"""Self-contained HTML/SVG. Numerical accounting stays in analyzer.py."""
from decimal import Decimal
from html import escape

from .analyzer import even_indices


def esc(value):
    return escape("Unavailable" if value is None else str(value), quote=True)


def display_rows(rows, important, limit=200):
    selected = even_indices(len(rows), limit - len(important))
    selected.update(i for i, row in enumerate(rows) if row["index"] in important)
    return [row for i, row in enumerate(rows) if i in selected]


def chart(rows, value, title, limits=()):
    values = [float(value(row)) for row in rows if value(row) is not None]
    values.extend(limits)
    if not values:
        return '<p class="muted">No valid values to plot.</p>'
    low, high = min(values), max(values)
    pad = (high - low) * 0.1 or 1
    low, high = low - pad, high + pad
    width, height = 900, 190
    last = rows[-1]["index"] or 1
    x = lambda row: 75 + row["index"] / last * 795
    y = lambda v: 15 + (high - float(v)) / (high - low) * 140
    chunks, segment = [], []
    for row in rows:
        v = value(row)
        if v is None:
            if segment:
                chunks.append(segment)
                segment = []
        else:
            segment.append(f"{x(row):.2f},{y(v):.2f}")
    if segment:
        chunks.append(segment)
    lines = ''.join(f'<polyline points="{" ".join(points)}" fill="none" stroke="#0f766e" stroke-width="2.5"/>' for points in chunks)
    for v in limits:
        lines += f'<line x1="75" x2="870" y1="{y(v):.2f}" y2="{y(v):.2f}" stroke="#b45309" stroke-dasharray="5 4"/>'
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
            f'<title>{esc(title)}</title><text x="4" y="23">{high:.2f}</text><text x="4" y="153">{low:.2f}</text>'
            f'{lines}<text x="75" y="183">Event order 0</text><text x="770" y="183">{last}</text></svg>')


def render(report):
    summary = report["summary"]
    imported = report.get("import_provenance")
    quote_only = bool(imported and imported["mode"] == "quotes_only")
    import_section = ""
    if imported:
        import_section = (f'<section id="import"><h2>Raw-log import and declared assumptions</h2>'
                          f'<p>Community format {esc(imported["source_format"])} at declared revision <code>{esc(imported["declared_producer_revision"])}</code>. '
                          f'Edition {esc(imported["edition"])}; round {esc(imported["round"])}; day {esc(imported["day"])}; mode {esc(imported["mode"])}.</p>'
                          f'<p>{imported["activity_rows"]} activity rows; {imported["own_trade_rows"]} own trade rows; '
                          f'{imported["market_trade_rows_excluded"]} market trade rows excluded; {imported["own_trade_rows_excluded"]} own rows excluded by mode.</p>'
                          f'<p>Declared self identity: {esc(imported["self_id"])}. Sequence policy: {esc(imported["sequence_policy"])}. '
                          f'Fee per own fill: {esc(imported["fee_policy"]["per_fill"])} XIREC; basis: {esc(imported["fee_policy"]["basis"])}.</p>'
                          f'<p>{esc(imported["limitations"])}</p>'
                          f'<p>{esc("Quote-only diagnostic: the explicitly empty portfolio has no strategy P&L; zero cash/equity values are empty-book conventions." if quote_only else "Accounting reflects the declared opening state and fee policy; it does not reproduce source-reported P&L or official matching.")}</p>'
                          f'<p>Raw source SHA-256: <code>{esc(imported["raw_source_sha256"])}</code><br>'
                          f'Import config SHA-256: <code>{esc(imported["config_sha256"])}</code><br>'
                          f'Normalized input SHA-256: <code>{esc(imported["normalized_sha256"])}</code></p>'
                          '<p><a href="raw-source.log" download>Raw log</a> · <a href="import-config.json">Import configuration</a> · '
                          '<a href="normalized.jsonl" download>Normalized events</a> · <a href="import-receipt.json">Import receipt</a></p></section>')
    important = {v for v in (summary["first_breach_index"], summary["first_unavailable_valuation_index"]) if v is not None}
    rows = display_rows(report["snapshots"], important)
    selected = even_indices(len(report["fills"]), 199)
    for i, fill in enumerate(report["fills"]):
        gross = fill["markout"]["gross_total"]
        if gross is not None and Decimal(gross) < 0:
            selected.add(i)
            break
    fills = [f for i, f in enumerate(report["fills"]) if i in selected]
    alerts = []
    if summary["first_breach_index"] is not None:
        alerts.append(f'<a href="#event-{summary["first_breach_index"]}">Inspect first inventory-limit breach</a>')
    if summary["first_unavailable_valuation_index"] is not None:
        alerts.append(f'<a href="#event-{summary["first_unavailable_valuation_index"]}">Inspect first missing/stale valuation</a>')
    if summary["adverse_markout_fills"]:
        alerts.append('<a href="#fills">Inspect negative fill markouts</a>')
    card = lambda title, value, note: f'<div class="card"><span>{esc(title)}</span><strong>{esc(value)}</strong><small>{esc(note)}</small></div>'
    cards = ''.join([
        card("Strategy P&L" if quote_only else "Marked P&L", "Not evaluated" if quote_only else summary["pnl"], "Quote-only import" if quote_only else report["currency"] + " · after declared fees"),
        card("Cash / equity", f'{summary["cash"]} / {summary["equity"] or "Unavailable"}', "Exact decimal accounting"),
        card("Limit breach observations", summary["breach_observations"], "Per instrument × event; not distinct episodes"),
        card("Adverse fill markouts", summary["adverse_markout_fills"], "Negative signed gross markout; not strategy P&L")])
    ledger = []
    for row in rows:
        positions = '; '.join(f'{s}: {q}' for s, q in row["positions"].items())
        quotes = '; '.join(f'{s}: mark {v["mark"]}, bid/ask {v["bid"]}/{v["ask"]}, spread {v["spread"]}, age {v["age"]} ({v["status"]})' for s, v in row["marks"].items())
        status = []
        if row["limit_breaches"]:
            status.append("LIMIT BREACH: " + '; '.join(f'{b["instrument"]} {b["position"]} vs ±{b["limit"]}' for b in row["limit_breaches"]))
        if row["valuation_issues"]:
            status.append("VALUATION UNAVAILABLE: " + '; '.join(f'{v["instrument"]} {v["status"]}' for v in row["valuation_issues"]))
        description = f'{row["type"]} {row["fill_id"] or row["instrument"] or "opening"}'
        ledger.append(f'<tr id="event-{row["index"]}" class="{"flag" if status else ""}">'
                      f'<td>{row["index"]}<br><small>{row["timestamp"]}:{row["sequence"]}</small></td>'
                      f'<td>{esc(description)}<small>{esc(row["note"])}</small><b>{esc("; ".join(status))}</b></td>'
                      f'<td>{esc(row["cash"])}<small>Δ {esc(row["cash_change"])}</small></td>'
                      f'<td>{esc(positions)}</td><td>{esc(row["position_value"])}<small>Δ {esc(row["position_value_change"])}</small></td>'
                      f'<td>{esc(row["equity"])}<small>Δ {esc(row["equity_change"])}</small></td><td>{esc(row["pnl"])}</td>'
                      f'<td><details><summary>Marks & freshness</summary>{esc(quotes)}<p>Last-known equity (may be stale): {esc(row["last_known_equity"])}</p></details></td></tr>')
    fill_rows = []
    for fill in fills:
        markout = fill["markout"]
        adverse = markout["gross_total"] is not None and Decimal(markout["gross_total"]) < 0
        fill_rows.append(f'<tr class="{"flag" if adverse else ""}"><td>{esc(fill["fill_id"])}<small>{esc(fill["note"])}</small></td>'
                         f'<td>{esc(fill["instrument"])}<br>{esc(fill["side"])} {fill["units"]} @ {esc(fill["price"])}<small>Fee {esc(fill["fee"])}</small></td>'
                         f'<td>{fill["timestamp"]}:{fill["sequence"]}<small>Target {markout["target_timestamp"]}</small></td>'
                         f'<td>{esc(fill["at_fill_mark"]["mark"])}<small>{esc(fill["at_fill_mark"]["status"])}</small></td>'
                         f'<td>{esc(markout["mark"])}<small>Observed {esc(markout["mark_timestamp"])}:{esc(markout["mark_sequence"])}; age {esc(markout["mark_age"])}</small></td>'
                         f'<td>{esc(markout["per_unit"])} / {esc(markout["gross_total"])}</td><td>{esc(markout["status"])}</td></tr>')
    inventory = ''.join(f'<h3>{esc(s)} · units, limit ±{spec["limit"]}</h3>'+chart(rows, lambda row, s=s: row["positions"][s], f"{s} inventory", [-spec["limit"], spec["limit"]]) for s, spec in report["instruments"].items())
    sampling = (f'HTML displays {len(rows)} of {report["details"]["total_snapshots"]} snapshots and {len(fills)} of {summary["fill_count"]} fills. '
                f'JSON detail mode: {report["details"]["mode"]}, retaining {report["details"]["retained_snapshots"]} snapshots. '
                'Exact summary counts and accounting use every event. Charts/tables may omit intermediate extrema and anomalies; first breach and first unavailable valuation are retained. Use bounded full JSON for raw drill-down.')
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>{esc(report["label"])} · IMC analysis</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7f8;color:#132d39;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px 24px}}h1{{font-size:clamp(1.9rem,4vw,3rem);line-height:1.12;margin:16px 0}}h2{{margin-top:32px}}h3{{margin-bottom:0}}a{{color:#075e65}}.eyebrow{{font-size:12px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#0f766e}}.muted,small{{color:#506673}}small{{display:block;font-size:12px;margin-top:4px}}.banner{{padding:14px 18px;background:#fff3d9;border-left:4px solid #b45309;border-radius:4px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}.card{{padding:18px;background:white;border:1px solid #dce6e8;border-radius:12px}}.card span{{font-size:13px}}.card strong{{display:block;font-size:25px;margin-top:8px}}section{{background:white;border:1px solid #dce6e8;padding:22px;border-radius:12px;margin:20px 0}}nav,.alerts{{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}}.alerts a{{padding:8px 12px;background:#fff3d9;border-radius:6px}}svg{{width:100%;height:auto;display:block}}svg text{{font:11px system-ui;fill:#506673}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;min-width:820px}}th,td{{padding:12px 10px;border-bottom:1px solid #e4ebed;text-align:left;vertical-align:top}}th{{background:#f0f5f6}}.flag{{background:#fff6e7}}td b{{display:block;font-size:11px;color:#9a3412;margin-top:5px}}tr:target{{outline:3px solid #b45309;outline-offset:-3px}}details{{max-width:340px}}summary{{cursor:pointer;color:#075e65}}code{{overflow-wrap:anywhere;font-size:12px}}footer{{padding:16px 0;color:#506673}}@media(max-width:700px){{main{{padding:20px 12px}}.cards{{grid-template-columns:repeat(2,1fr)}}section{{padding:16px}}.card{{padding:12px}}.card strong{{font-size:21px}}}}@media(max-width:400px){{.cards{{grid-template-columns:1fr}}}}
</style></head><body><main>
<div class="eyebrow">Offline research workbench · {esc(report["data_kind"])} data</div>
<h1>{esc(report["label"])}</h1>
<p class="muted">Explain fills, inventory and changes in marked equity. Follow the highlighted records to investigate a costly fill or limit excursion.</p>
<div class="banner">{esc(report["provenance"])} Synthetic demonstrations are not market-performance evidence.</div>
<nav aria-label="Report sections"><a href="#equity">Equity</a><a href="#inventory">Inventory</a><a href="#fills">Fill quality</a><a href="#ledger">Event ledger</a><a href="#contract">Contract</a></nav>
<div class="cards">{cards}</div><div class="alerts">{''.join(alerts) or 'No breach or adverse-markout flags in this run.'}</div>
<p class="muted">{esc(sampling)}</p>
{import_section}
<section id="equity"><h2>Marked equity</h2><p>Opening equity {esc(report["opening_equity"])} {esc(report["currency"])}. Missing or stale marks on held instruments create gaps; later observations never revise earlier valuations.</p>{chart(rows, lambda r:r["equity"], "Marked equity by event order")}</section>
<section id="inventory"><h2>Inventory and declared limits</h2><p>Dashed lines show each instrument's absolute position limit. Filled positions are analyzed; this tool does not simulate orders.</p>{inventory}</section>
<section id="fills"><h2>Fill quality at +{report["markout_horizon"]} {esc(report["timestamp_unit"])}</h2><p>Signed gross markout = (target mark − fill price) × side × units. Buy side is +1; sell side is −1. Negative means the fill price was unfavorable relative to the horizon mark; it does not by itself establish a causal adverse-selection mechanism. Fees are shown separately and are included in cash P&L. Future markouts are retrospective diagnostics.</p><div class="scroll"><table><thead><tr><th>Fill / note</th><th>Execution</th><th>Time / horizon</th><th>At-fill mark</th><th>Target as-of mark</th><th>Per unit / gross total</th><th>Status</th></tr></thead><tbody>{''.join(fill_rows)}</tbody></table></div></section>
<section id="ledger"><h2>Event ledger and valuation changes</h2><p>Δ columns compare each event with its actual preceding event, even when rows are omitted here. Cash change + marked-position-value change = equity change when both valuations are valid.</p><div class="scroll"><table><thead><tr><th>Event / time:seq</th><th>Event and diagnostic</th><th>Cash / Δ</th><th>Positions</th><th>Position value / Δ</th><th>Equity / Δ</th><th>P&L</th><th>Quote details</th></tr></thead><tbody>{''.join(ledger)}</tbody></table></div></section>
<section id="contract"><h2>Contract and evidence</h2><p>All prices are {esc(report["currency"])} per integer instrument unit. Time is an opaque {esc(report["timestamp_unit"])} clock, not UTC. Events are strictly ordered by timestamp then sequence; quotes become available at that event. Fill time records execution availability, not an inferred decision timestamp. Mark freshness allows age ≤{report["mark_max_age"]}.</p><p>Horizon marks use the last observation at or before the target, with no interpolation. A carried mark within the freshness threshold may predate the fill; its actual timestamp is shown. Targets beyond run end are unavailable. No realized/unrealized lot decomposition, conversions, financing, FX or matching model is implemented.</p><p>Exact input: {report["source_bytes"]} bytes · {summary["event_count"]} events · {summary["fill_count"]} fills<br>SHA-256: <code>{esc(report["source_sha256"])}</code><br>Analyzer version {esc(report["tool_version"])} · report schema {esc(report["schema"])}<br>Installed Python source fingerprint: <code>{esc(report["code_identity"]["sha256"])}</code><br>{esc(report["code_identity"]["scope"])}<br>Runtime: {esc(report["runtime"]["implementation"])} {esc(report["runtime"]["python_version"])}</p><p><a href="report.json">Numerical JSON</a> · <a href="complete.json">Completion hashes</a></p></section>
<footer>New post-competition analysis tooling. No trades, network services or historical team-result claims.</footer>
</main></body></html>'''
