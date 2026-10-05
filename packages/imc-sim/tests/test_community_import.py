from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from imc4_analysis.analyzer import analyze
from imc4_analysis.cli import main, write_report
from imc4_analysis.community_import import normalize_community_log, _remove_object_trailing_commas
from imc4_analysis.contracts import InputError
from imc4_analysis.report import render

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/community-log"


def inputs():
    return (EXAMPLES/"sample.log").read_bytes(), (EXAMPLES/"import.json").read_bytes()


def configured(**changes):
    raw, configuration = inputs()
    config = json.loads(configuration)
    config.update(changes)
    return raw, json.dumps(config).encode()


def with_trades(raw, update):
    head, text = raw.decode().split("Trade History:\n")
    trades = json.loads(_remove_object_trailing_commas(text))
    update(trades)
    return (head+"Trade History:\n"+json.dumps(trades)).encode()


class CommunityImportTests(unittest.TestCase):
    def test_hand_ledger_and_market_exclusion(self):
        raw, config = inputs()
        normalized, provenance = normalize_community_log(raw, config)
        report = analyze(normalized)
        self.assertEqual(report["summary"]["cash"], "901")
        self.assertEqual(report["summary"]["positions"], {"ALPHA":1,"BETA":0})
        self.assertEqual(report["summary"]["equity"], "1003")
        self.assertEqual(report["summary"]["pnl"], "3")
        self.assertEqual(report["summary"]["total_fees"], "4")
        self.assertEqual(provenance["market_trade_rows_excluded"], 2)
        self.assertEqual(provenance["own_trade_rows"], 4)
        self.assertEqual([f["markout"]["gross_total"] for f in report["fills"]], ["6","2","3",None])
        self.assertEqual(report["summary"]["first_breach_index"], 3)

    def test_timestamp_ties_and_source_order(self):
        normalized, _ = normalize_community_log(*inputs())
        rows = [json.loads(line) for line in normalized.splitlines()][1:]
        at_zero = [r for r in rows if r["timestamp"] == 0]
        self.assertEqual([r["sequence"] for r in at_zero], [0,1,2,3])
        self.assertEqual([(r["type"],r["instrument"]) for r in at_zero], [("quote","ALPHA"),("quote","BETA"),("fill","ALPHA"),("fill","BETA")])
        self.assertEqual(at_zero[3]["side"], "sell")

    def test_future_rows_do_not_change_earlier_valuations(self):
        raw, config = inputs()
        earlier = with_trades(raw, lambda rows: rows.__setitem__(slice(None), [r for r in rows if r["timestamp"] <= 100]))
        earlier = b"\n".join(line for line in earlier.split(b"\n") if not line.startswith(b"0;200;"))
        short, _ = normalize_community_log(earlier, config)
        complete, _ = normalize_community_log(raw, config)
        short_report, complete_report = analyze(short, detail="full"), analyze(complete, detail="full")
        def valuations(report):
            # Record IDs bind the whole raw file and therefore change on append.
            return [{k:v for k,v in row.items() if k != "fill_id"}
                    for row in report["snapshots"] if row["timestamp"] <= 100]
        self.assertEqual(valuations(short_report), valuations(complete_report))
        self.assertEqual(short_report["fills"][-1]["markout"]["status"], "unavailable_future")
        self.assertEqual(complete_report["fills"][2]["markout"]["gross_total"], "3")

    def test_commitments_bind_raw_config_and_normalized_bytes(self):
        raw, config = inputs()
        normalized, p = normalize_community_log(raw, config)
        self.assertEqual(p["raw_source_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(p["config_sha256"], hashlib.sha256(config).hexdigest())
        self.assertEqual(p["normalized_sha256"], hashlib.sha256(normalized).hexdigest())
        altered, q = normalize_community_log(raw+b"\n", config)
        self.assertNotEqual(p["raw_source_sha256"], q["raw_source_sha256"])
        self.assertNotEqual(p["normalized_sha256"], q["normalized_sha256"])
        self.assertEqual(analyze(normalized)["summary"], analyze(altered)["summary"])
        _, c = normalize_community_log(raw, config+b"\n")
        self.assertNotEqual(p["config_sha256"], c["config_sha256"])

    def test_missing_declarations_and_wrong_edition_units_refused(self):
        raw, config = inputs()
        for field in ("self_id","sequence_policy","fee_policy","fill_identity_policy"):
            changed = json.loads(config)
            del changed[field]
            with self.subTest(missing=field), self.assertRaises(InputError):
                normalize_community_log(raw,json.dumps(changed).encode())
        for field,value in (("edition","prosperity3"),("producer_revision","latest"),("quantity_unit","lots"),("sequence_policy","guess"),("self_id",""),("day",True)):
            with self.subTest(field=field), self.assertRaises(InputError):
                normalize_community_log(*configured(**{field:value}))
        for field,value in (("currency","USD"),("timestamp_unit","UTC_seconds")):
            changed=json.loads(config);changed["analysis"][field]=value
            with self.subTest(field=field), self.assertRaises(InputError):
                normalize_community_log(raw,json.dumps(changed).encode())

    def test_duplicate_own_records_refused_even_numeric_spelling_differs(self):
        raw,config=inputs()
        def duplicate(rows):
            other=copy.deepcopy(rows[0]);other["price"]="101.0";rows.insert(1,other)
        with self.assertRaisesRegex(InputError,"indistinguishable duplicate"):
            normalize_community_log(with_trades(raw,duplicate),config)

    def test_ambiguous_side_identity_and_malformed_trade_refused(self):
        raw,config=inputs()
        for update in (lambda r:r[0].update(seller="SUBMISSION"),lambda r:r[0].update(buyer=1),
                       lambda r:r[0].update(quantity=0),lambda r:r[0].update(currency="SEASHELLS"),
                       lambda r:r[0].update(symbol=[]),lambda r:r[0].pop("price"),
                       lambda r:r[0].update(side="buy")):
            with self.subTest(update=update), self.assertRaises(InputError):
                normalize_community_log(with_trades(raw,update),config)

    def test_market_only_requires_explicit_quote_mode(self):
        raw,config=inputs()
        market=with_trades(raw,lambda r:r.__setitem__(slice(None),[r[1],r[4]]))
        with self.assertRaisesRegex(InputError,"no own fills"):
            normalize_community_log(market,config)
        config=json.loads(config);config["mode"]="quotes_only";config["analysis"]["initial_cash"]="0"
        normalized,p=normalize_community_log(market,json.dumps(config).encode())
        report=analyze(normalized);report["import_provenance"]=p
        self.assertEqual(report["summary"]["fill_count"],0)
        self.assertFalse(p["strategy_pnl_available"])
        self.assertIn("Not evaluated",render(report))
        self.assertIn("no strategy P&amp;L",render(report))

    def test_quote_mode_cannot_invent_opening_portfolio(self):
        with self.assertRaisesRegex(InputError,"zero cash"):
            normalize_community_log(*configured(mode="quotes_only"))

    def test_partial_activity_and_multiple_days_refused(self):
        raw,config=inputs()
        for bad in (raw.replace(b"0;0;ALPHA",b"1;0;ALPHA",1),
                    raw.replace(b"99;10;",b"99;;",1),raw.replace(b";100;0\n",b";100\n",1),
                    raw.replace(b"day;timestamp",b"day;time")):
            with self.subTest(raw=bad[:40]), self.assertRaises(InputError):
                normalize_community_log(bad,config)

    def test_out_of_order_sections_refused(self):
        raw,config=inputs()
        changed=with_trades(raw,lambda r:r.reverse())
        with self.assertRaisesRegex(InputError,"timestamp-ordered"):
            normalize_community_log(changed,config)
        changed=raw.replace(b"0;100;ALPHA",b"0;300;ALPHA")
        with self.assertRaisesRegex(InputError,"timestamp-ordered"):
            normalize_community_log(changed,config)

    def test_trailing_comma_handling_never_changes_quoted_text(self):
        value='[{"buyer":"literal , } text","seller":"escaped \\\" , }",}]'
        parsed=json.loads(_remove_object_trailing_commas(value))
        self.assertEqual(parsed[0]["buyer"],"literal , } text")
        self.assertEqual(parsed[0]["seller"],'escaped " , }')

    def test_cli_import_bundle_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"imported"
            with redirect_stdout(io.StringIO()):
                result=main(["import-log",str(EXAMPLES/"sample.log"),"--config",str(EXAMPLES/"import.json"),"--out",str(target)])
            self.assertEqual(result,0)
            receipt=json.loads((target/"complete.json").read_text())
            report=json.loads((target/"report.json").read_text())
            self.assertEqual(len(receipt["files"]),6)
            self.assertEqual(report["source_sha256"],report["import_provenance"]["normalized_sha256"])
            self.assertEqual(receipt["import_provenance"],report["import_provenance"])
            for name,details in receipt["files"].items():
                self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),details["sha256"])
            self.assertEqual((target/"raw-source.log").read_bytes(),inputs()[0])
            self.assertEqual((target/"import-config.json").read_bytes(),inputs()[1])

    def test_wrong_import_commitment_refused_before_output(self):
        raw,config=inputs();normalized,p=normalize_community_log(raw,config)
        report=analyze(normalized);report["import_provenance"]=p
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"bad"
            with self.assertRaisesRegex(InputError,"commitment mismatch"):
                write_report(report,target,{"raw-source.log":raw+b"x","import-config.json":config,"normalized.jsonl":normalized,"import-receipt.json":b"{}"})
            self.assertFalse(target.exists())
