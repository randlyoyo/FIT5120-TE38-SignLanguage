#!/usr/bin/env python3
"""Build a self-contained review page for a directory of comparison GIFs.

    python make_index.py renders/review

Written for handing to someone else: everything is relative paths and embedded
data, so the folder works unzipped on any machine with a browser and no server,
no network and nothing installed.

Paged on purpose. All 3,215 animations on one page is well over a gigabyte of
decoded image, and the browser stalls long before a reviewer gets anywhere.
"""
import argparse
import html
import json
from pathlib import Path

PAGE = """<!doctype html><meta charset="utf-8"><title>__TITLE__</title>
<style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#dfe4ea;--ink:#161b24;--ink2:#5d6672;--flag:#c2453f;--acc:#2f8fd0}
@media(prefers-color-scheme:dark){:root{--bg:#0f131a;--panel:#161b24;--line:#252c38;--ink:#e6ebf2;--ink2:#98a2b1}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
header{position:sticky;top:0;background:var(--panel);border-bottom:1px solid var(--line);
 padding:11px 18px;display:flex;gap:11px;align-items:center;flex-wrap:wrap;z-index:9}
h1{margin:0;font-size:17px}
input,button,select{font:inherit;font-size:13px;padding:6px 10px;border:1px solid var(--line);
 border-radius:6px;background:var(--bg);color:var(--ink)}
button{cursor:pointer}
button[aria-pressed=true]{background:var(--ink);color:var(--panel);border-color:var(--ink)}
.n{margin-left:auto;color:var(--ink2);font-size:12.5px;font-family:ui-monospace,Menlo,monospace}
.hint{padding:12px 18px 0;color:var(--ink2);font-size:12.5px;max-width:96ch;line-height:1.5}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(430px,1fr));gap:14px;padding:14px 18px 10px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:9px;
 overflow:hidden;border-left:3px solid var(--line)}
figure.flag{border-left-color:var(--flag)}
img{display:block;width:100%;background:#fff;min-height:150px}
figcaption{padding:7px 11px 9px}
figcaption b{font-size:13px;display:block}
figcaption span{font-size:11.5px;color:var(--ink2);font-family:ui-monospace,Menlo,monospace}
figcaption i{color:var(--flag);font-style:normal}
.pg{display:flex;gap:9px;justify-content:center;align-items:center;padding:6px 0 34px}
.pg button{padding:6px 14px;background:var(--panel)}
.pg button[disabled]{opacity:.4;cursor:default}
</style>
<header>
 <h1>__TITLE__</h1>
 <input type="search" id="q" placeholder="Search a sign&hellip;">
 <button id="fFlag" aria-pressed="false">Flagged by check</button>
 <button id="fAbs" aria-pressed="false">Has absent hands</button>
 <select id="per"><option>24</option><option selected>48</option><option>96</option></select>
 <span class="n" id="n"></span>
</header>
<p class="hint">__HINT__</p>
<main id="g"></main>
<div class="pg"><button id="prev">Previous</button><span class="n" id="pg"></span><button id="next">Next</button></div>
<script id="d" type="application/json">__DATA__</script>
<script>
var ROWS=JSON.parse(document.getElementById('d').textContent);
var q=document.getElementById('q'),fF=document.getElementById('fFlag'),
    fA=document.getElementById('fAbs'),per=document.getElementById('per'),
    g=document.getElementById('g'),n=document.getElementById('n'),pg=document.getElementById('pg');
var view=ROWS,page=0;
function apply(){
  var s=q.value.trim().toUpperCase();
  var oF=fF.getAttribute('aria-pressed')==='true', oA=fA.getAttribute('aria-pressed')==='true';
  view=ROWS.filter(function(r){
    if(s&&r[1].toUpperCase().indexOf(s)<0) return false;
    if(oF&&!r[5].length) return false;
    if(oA&&!(r[3]>0.005)) return false;
    return true;});
  page=0; render();
}
function render(){
  var P=+per.value, st=page*P, items=view.slice(st,st+P);
  g.textContent='';
  items.forEach(function(r){
    var f=document.createElement('figure'); if(r[5].length) f.className='flag';
    var i=document.createElement('img'); i.loading='lazy'; i.src=encodeURI(r[0]); i.alt=r[1];
    var c=document.createElement('figcaption');
    var b=document.createElement('b'); b.textContent=r[1];
    var s2=document.createElement('span');
    s2.textContent=r[2].toFixed(2)+' · '+r[4]+(r[3]>0.005?' · '+Math.round(r[3]*100)+'% absent':'');
    if(r[5].length){var em=document.createElement('i'); em.textContent=' · '+r[5].join(', '); s2.appendChild(em);}
    c.appendChild(b); c.appendChild(s2);
    f.appendChild(i); f.appendChild(c); g.appendChild(f);
  });
  n.textContent=view.length+' / '+ROWS.length;
  pg.textContent=(page+1)+' / '+Math.max(1,Math.ceil(view.length/P));
  document.getElementById('prev').disabled=page===0;
  document.getElementById('next').disabled=st+P>=view.length;
  window.scrollTo(0,0);
}
q.addEventListener('input',apply); per.addEventListener('change',apply);
[fF,fA].forEach(function(b){b.addEventListener('click',function(){
  b.setAttribute('aria-pressed',b.getAttribute('aria-pressed')==='true'?'false':'true');apply();});});
document.getElementById('prev').addEventListener('click',function(){if(page>0){page--;render();}});
document.getElementById('next').addEventListener('click',function(){
  if((page+1)*(+per.value)<view.length){page++;render();}});
apply();
</script>"""

HINT = ("Left is the reconstructed rig, middle the same head enlarged, right the source "
        "video &mdash; all three are the same frame. Sorted with the most suspect first, so "
        "you can stop once the problems stop appearing. Two things are worth checking: the "
        "HANDS (shape, orientation, whether the movement matches the video) and the FACE "
        "(brow shape especially &mdash; an inner-raise and a full lower are different "
        "grammatical markers in Auslan). Do not open a single GIF in macOS Preview: it lists "
        "the frames instead of playing them. Use a browser, or press space in Finder.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--title", default="Rig vs video")
    a = ap.parse_args()
    D = Path(a.folder)
    man = {w["word"]: w for w in json.load(open("renders/manifest.json"))["words"]}
    rc = json.load(open("renders/render_check.json"))
    chk = {r["word"]: r for r in rc["results"]}
    th = rc["thresholds"]
    rows = []
    for p in sorted(D.glob("*.gif")):
        w = p.stem
        m, c = man.get(w, {}), chk.get(w, {})
        fl = [k for k in ("visFlicker", "jump", "dirFlip")
              if c.get(k, 0) > th.get(k, 1e9)]
        rows.append([p.name, w, round(m.get("score", 0), 2),
                     round(m.get("absent", 0), 4),
                     f"{m.get('split','?')}/{m.get('stem','?').replace('_kf_rgb','')}",
                     fl])
    rows.sort(key=lambda r: (-len(r[5]), -r[3], r[2]))
    page = (PAGE.replace("__TITLE__", html.escape(a.title))
                .replace("__HINT__", HINT)
                .replace("__DATA__", json.dumps(rows, separators=(",", ":"))))
    (D / "index.html").write_text(page)
    print(f"{len(rows)} signs, {sum(1 for r in rows if r[5])} flagged -> {D/'index.html'}")


if __name__ == "__main__":
    main()
