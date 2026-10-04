// Python view of the C++ replay, built by `make ext` into build/python/.
//
// It takes the reference's own inputs (keyframes as (ts, [Level]) with Level
// exposing .side/.tick/.size_e2, deltas as 4-item rows, coverage as
// (start, end | None)) and answers with each book's sorted (side, tick, size_e2)
// tuples. replay.book.reconstruct_many(..., native=True) wraps it into Books.
//
// Errors: unsorted history raises ValueError and refusals raise this module's
// NotCovered, both with the reference's message text. Inputs C++ cannot hold
// exactly are rejected, never coerced: a non-int (float or NaN query time,
// say) raises TypeError, and an int outside int64 (int32 for ticks) or a side
// other than 0/1 raises ValueError. The Python reference accepts some of those,
// which is why the native path is opt-in.
//
// Row conversion uses the CPython API directly: it is the bulk of the cost of
// crossing from Python, since the replay itself is about a millisecond.
#include "asof/replay/replay.hpp"

#include <pybind11/pybind11.h>

#include <Python.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace asof::replay;

namespace {

std::int64_t to_i64(PyObject* value, const char* what) {
    if (!PyLong_Check(value)) {
        throw py::type_error(std::string(what) + " must be an int, not " + Py_TYPE(value)->tp_name);
    }
    int overflow = 0;
    const long long out = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (overflow != 0) throw py::value_error(std::string(what) + " is outside the int64 range");
    if (out == -1 && PyErr_Occurred()) throw py::error_already_set();
    return out;
}

std::int32_t to_tick(PyObject* value) {
    const std::int64_t tick = to_i64(value, "tick");
    if (tick < std::numeric_limits<std::int32_t>::min() || tick > std::numeric_limits<std::int32_t>::max()) {
        throw py::value_error("tick is outside the int32 range");
    }
    return static_cast<std::int32_t>(tick);
}

Side to_side(PyObject* value) {
    const std::int64_t side = to_i64(value, "side");
    if (side != 0 && side != 1) throw py::value_error("side must be 0 (bid) or 1 (ask)");
    return static_cast<Side>(side);
}

// Any iterable, materialised once as a list or tuple (no copy if it already is one).
py::object fast_sequence(py::handle rows, const char* what) {
    PyObject* seq = PySequence_Fast(rows.ptr(), what);
    if (seq == nullptr) throw py::error_already_set();
    return py::reinterpret_steal<py::object>(seq);
}

// Calls fn on each item of a list or tuple. The size is re-read every step and
// each item is held by a strong reference while fn runs: Python code that runs
// meanwhile (a property getter, or a finalizer the garbage collector triggers)
// can shrink a list and free its items, and a cached size or item array would
// then be read after free.
template <class Fn>
void for_each_item(const py::object& seq, Fn&& fn) {
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(seq.ptr()); ++i) {
        fn(py::reinterpret_borrow<py::object>(PySequence_Fast_GET_ITEM(seq.ptr(), i)));
    }
}

// One row of a fixed width, e.g. a (ts, side, tick, size) delta, as a tuple
// (a copy unless it already is one), so its fields stay alive and fixed
// whatever Python code runs later.
py::tuple row_of(py::object row, Py_ssize_t width, const char* what) {
    if (!PyTuple_CheckExact(row.ptr())) {
        PyObject* tuple = PySequence_Tuple(row.ptr());
        if (tuple == nullptr) throw py::error_already_set();
        row = py::reinterpret_steal<py::object>(tuple);
    }
    if (PyTuple_GET_SIZE(row.ptr()) != width) {
        throw py::value_error(std::string(what) + " must have " + std::to_string(width) + " items");
    }
    return py::reinterpret_steal<py::tuple>(row.release());  // hand on the reference we hold
}

std::vector<std::int64_t> to_times(py::handle rows) {
    std::vector<std::int64_t> out;
    for_each_item(fast_sequence(rows, "at_times must be iterable"),
                  [&](py::object q) { out.push_back(to_i64(q.ptr(), "query time")); });
    return out;
}

std::vector<Keyframe> to_keyframes(const py::tuple& frames) {
    std::vector<Keyframe> out;
    out.reserve(frames.size());
    for_each_item(frames, [&](py::object row) {
        const py::tuple pair = row_of(std::move(row), 2, "a keyframe row must be (ts_ms, levels)");
        Keyframe frame{to_i64(PyTuple_GET_ITEM(pair.ptr(), 0), "keyframe ts_ms"), {}};
        for (const py::handle level : py::reinterpret_borrow<py::iterable>(PyTuple_GET_ITEM(pair.ptr(), 1))) {
            frame.levels.push_back({to_side(level.attr("side").ptr()), to_tick(level.attr("tick").ptr()),
                                    to_i64(level.attr("size_e2").ptr(), "size_e2")});
        }
        out.push_back(std::move(frame));
    });
    return out;
}

std::vector<Delta> to_deltas(py::handle rows) {
    const py::object seq = fast_sequence(rows, "deltas must be iterable");
    std::vector<Delta> out;
    out.reserve(static_cast<std::size_t>(PySequence_Fast_GET_SIZE(seq.ptr())));
    for_each_item(seq, [&](py::object row) {
        const py::tuple f = row_of(std::move(row), 4, "a delta row must be (ts_ms, side, tick, size_e2)");
        out.push_back({to_i64(PyTuple_GET_ITEM(f.ptr(), 0), "delta ts_ms"), to_side(PyTuple_GET_ITEM(f.ptr(), 1)),
                       to_tick(PyTuple_GET_ITEM(f.ptr(), 2)), to_i64(PyTuple_GET_ITEM(f.ptr(), 3), "size_e2")});
    });
    return out;
}

std::vector<Interval> to_coverage(py::handle rows) {
    std::vector<Interval> out;
    for_each_item(fast_sequence(rows, "coverage must be iterable"), [&](py::object row) {
        const py::tuple f = row_of(std::move(row), 2, "a coverage row must be (start_ms, end_ms or None)");
        PyObject* end = PyTuple_GET_ITEM(f.ptr(), 1);
        out.push_back({to_i64(PyTuple_GET_ITEM(f.ptr(), 0), "coverage start_ms"),
                       end == Py_None ? std::nullopt : std::optional{to_i64(end, "coverage end_ms")}});
    });
    return out;
}

// Conversion order mirrors the reference, which copies all three histories
// (list(...)) before it reads a single level attribute: keyframes are copied
// first, and their levels, the only conversion that can run Python code, are
// read last. A getter that mutates the caller's lists then sees the same
// snapshot semantics as replay.book.
struct History {
    std::vector<Keyframe> frames;
    std::vector<Delta> deltas;
    std::vector<Interval> coverage;
};

History to_history(py::handle keyframes, py::handle deltas, py::handle coverage) {
    PyObject* frames = PySequence_Tuple(keyframes.ptr());
    if (frames == nullptr) throw py::error_already_set();
    const auto snapshot = py::reinterpret_steal<py::tuple>(frames);
    History out;
    out.deltas = to_deltas(deltas);
    out.coverage = to_coverage(coverage);
    out.frames = to_keyframes(snapshot);
    return out;
}

// The sorted (side, tick, size_e2) list the reference's sorted(book.multiset()) gives.
py::list sorted_levels(const Book& book) {
    py::list out(book.levels().size());
    std::size_t i = 0;
    for (const Level& lv : book.levels()) {
        out[i++] = py::make_tuple(static_cast<int>(lv.side), lv.tick, lv.size_e2);
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(asof_replay_native, m) {
    m.doc() = "C++20 order-book replay; the Python replay.book module remains the reference.";
    py::register_exception<NotCovered>(m, "NotCovered");

    m.def(
        "reconstruct",
        [](py::handle keyframes, py::handle deltas, py::handle at_ms, py::handle coverage) {
            const std::int64_t at = to_i64(at_ms.ptr(), "query time");
            const History h = to_history(keyframes, deltas, coverage);
            return sorted_levels(reconstruct(h.frames, h.deltas, at, h.coverage));
        },
        py::arg("keyframes"), py::arg("deltas"), py::arg("at_ms"), py::arg("coverage"),
        "One book's sorted (side, tick, size_e2) levels at at_ms, or NotCovered.");

    m.def(
        "reconstruct_many",
        [](py::handle keyframes, py::handle deltas, py::handle at_times, py::handle coverage) {
            py::list out;
            const auto queries = to_times(at_times);
            if (queries.empty()) return out;  // the reference never inspects history for no queries
            const History h = to_history(keyframes, deltas, coverage);
            std::vector<Book> books;
            {
                py::gil_scoped_release unlocked;  // pure C++ on owned vectors from here
                books = reconstruct_many(h.frames, h.deltas, queries, h.coverage);
            }
            for (const Book& book : books) out.append(sorted_levels(book));
            return out;
        },
        py::arg("keyframes"), py::arg("deltas"), py::arg("at_times"), py::arg("coverage"),
        "Sorted levels for every query, in request order; ValueError on unsorted history.");
}
