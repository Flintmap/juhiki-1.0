#!/usr/bin/env node
// ═══════════════════════════════════════════════════════════════
//  CellularGpt Multi-Book GitHub Trainer
//  • Cycles through books/book1.txt, books/book2.txt etc in order
//  • Picks up exactly where it left off inside each book
//  • Saves every 60s, exits code 2 on time limit (workflow restarts)
//  • When ALL books looped AND time target hit → creates "finished" file
//  • Shows param heatmap, steps/sec, loss sparkline live
// ═══════════════════════════════════════════════════════════════
"use strict";

const fs   = require("fs");
const path = require("path");

// ─── CLI ──────────────────────────────────────────────────────
const args = process.argv.slice(2);
const flags = {};
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith("--")) { flags[args[i].slice(2)] = args[i + 1]; i++; }
}

const LR          = parseFloat(flags.lr     || "0.01");
const HIDDEN      = parseInt(flags.hidden   || "128",  10);
const CTX         = parseInt(flags.ctx      || "32",   10);
const EMBED       = parseInt(flags.embed    || "16",   10);
const MODEL_FILE  = flags.model             || "model.json";
const STATE_FILE  = flags.state             || "train-state.json";
const BOOKS_DIR   = flags.books             || "books";
const SAVE_MS     = parseInt(flags.save     || "60",  10) * 1000;
const MAX_HOURS   = parseFloat(flags.hours  || "4");
const MAX_MS      = MAX_HOURS * 60 * 60 * 1000;
const TARGET_DAYS = parseFloat(flags.days   || "2");
const TARGET_MS   = TARGET_DAYS * 24 * 60 * 60 * 1000;
const FRAME_MS    = 40;
const VIZ_MS      = parseInt(flags.viz      || "300", 10) * 1000;

// ─── Colours ──────────────────────────────────────────────────
const C = {
  reset:"\x1b[0m",bold:"\x1b[1m",dim:"\x1b[2m",
  cyan:"\x1b[36m",green:"\x1b[32m",yellow:"\x1b[33m",
  red:"\x1b[31m",magenta:"\x1b[35m",blue:"\x1b[34m",white:"\x1b[37m",
};
const f  = (c,t) => `${C[c]}${t}${C.reset}`;
const bf = (c,t) => `${C.bold}${C[c]}${t}${C.reset}`;
function bar(pct,w=30){const n=Math.min(w,Math.round(pct*w));return f("cyan","█".repeat(n))+f("dim","░".repeat(w-n));}
function cl(){process.stdout.write("\r\x1b[K");}

// ─── Neuron heatmap ───────────────────────────────────────────
const HC=["\x1b[38;5;17m","\x1b[38;5;19m","\x1b[38;5;21m","\x1b[38;5;33m","\x1b[38;5;45m","\x1b[38;5;51m","\x1b[38;5;82m","\x1b[38;5;118m","\x1b[38;5;226m","\x1b[38;5;196m"];
const HH=[" ","·","▪","▫","◆","◈","●","◉","⬟","■"];
function hCell(v,mn,mx){
  if(mx===mn)return HC[0]+HH[0]+C.reset;
  const n=Math.max(0,Math.min(1,(Math.abs(v)-mn)/(mx-mn)));
  const i=Math.min(9,Math.floor(n*10));
  return HC[i]+HH[i]+C.reset;
}
function drawViz(NET,lossHist,sps,elapsed,bookName,bookIdx,totalBooks){
  if(!NET)return;
  const{W1,b1,W2,hiddenSize,embedDim,ctxLen,V}=NET;
  console.log();
  console.log(bf("magenta","  ╔══════════════════════════════════════════════════════╗"));
  console.log(bf("magenta","  ║")+bf("white","          🧠  NEURON ACTIVITY MAP                  ")+bf("magenta","║"));
  console.log(bf("magenta","  ╚══════════════════════════════════════════════════════╝"));
  // W1 heatmap
  const r1=Math.min(16,hiddenSize),c1=Math.min(32,embedDim*ctxLen);
  let mn=Infinity,mx=-Infinity;
  for(let r=0;r<r1;r++)for(let c=0;c<c1;c++){const v=W1.data[r*W1.cols+c];if(v<mn)mn=v;if(v>mx)mx=v;}
  console.log(f("cyan",`\n  W1 [${r1}×${c1} of ${hiddenSize}×${embedDim*ctxLen}]`));
  for(let r=0;r<r1;r++){process.stdout.write("  ");for(let c=0;c<c1;c++)process.stdout.write(hCell(W1.data[r*W1.cols+c],Math.abs(mn),Math.abs(mx)));process.stdout.write("\n");}
  // W2 heatmap
  const r2=Math.min(16,hiddenSize),c2=Math.min(32,V);
  let mn2=Infinity,mx2=-Infinity;
  for(let r=0;r<r2;r++)for(let c=0;c<c2;c++){const v=W2.data[r*W2.cols+c];if(v<mn2)mn2=v;if(v>mx2)mx2=v;}
  console.log(f("green",`\n  W2 [${r2}×${c2} of ${hiddenSize}×${V}]`));
  for(let r=0;r<r2;r++){process.stdout.write("  ");for(let c=0;c<c2;c++)process.stdout.write(hCell(W2.data[r*W2.cols+c],Math.abs(mn2),Math.abs(mx2)));process.stdout.write("\n");}
  // biases
  console.log(f("yellow",`\n  b1 biases [${hiddenSize}]`));
  process.stdout.write("  ");
  let bMn=Infinity,bMx=-Infinity;
  for(const v of b1.data){if(v<bMn)bMn=v;if(v>bMx)bMx=v;}
  for(let i=0;i<Math.min(64,hiddenSize);i++)process.stdout.write(hCell(b1.data[i],Math.abs(bMn),Math.abs(bMx)));
  process.stdout.write("\n");
  // loss sparkline
  if(lossHist.length>1){
    const lMx=Math.max(...lossHist),lMn=Math.min(...lossHist);
    const sp="▁▂▃▄▅▆▇█";
    console.log(f("magenta",`\n  loss [last ${lossHist.length} saves]`));
    process.stdout.write("  ");
    for(const v of lossHist){const n=lMx>lMn?(v-lMn)/(lMx-lMn):0;const i=Math.min(7,Math.floor(n*8));const col=n>0.7?C.red:n>0.4?C.yellow:C.green;process.stdout.write(col+sp[i]+C.reset);}
    process.stdout.write(f("dim",`  ${lMn.toFixed(4)}→${lMx.toFixed(4)}\n`));
  }
  // stats
  const em=(elapsed/60000).toFixed(1),rm=((MAX_MS-elapsed)/60000).toFixed(1);
  const td=(elapsed/TARGET_MS*100).toFixed(1);
  console.log();
  console.log(f("cyan","  steps/s : ")+bf("white",sps.toLocaleString())+f("cyan","   total : ")+bf("white",trainSteps.toLocaleString())+f("cyan","   vocab : ")+bf("white",String(vocab.idToChar.length)));
  console.log(f("green","  elapsed : ")+bf("white",em+"min")+f("green","   left : ")+bf("white",rm+"min")+f("green","   loss : ")+bf("yellow",recentLoss!=null?recentLoss.toFixed(5):"—"));
  console.log(f("blue","  book    : ")+bf("white",`[${bookIdx+1}/${totalBooks}] ${bookName}`)+f("blue","   target: ")+bf("white",td+"% of "+TARGET_DAYS+"d"));
  process.stdout.write("  legend: ");for(let i=0;i<10;i++)process.stdout.write(HC[i]+HH[i]+C.reset);process.stdout.write(f("dim","  cold→hot\n\n"));
}

// ══════════════════════════════════════════════════════════════
// VOCAB
// ══════════════════════════════════════════════════════════════
const SPECIAL={PAD:0,UNK:1,BOS:2,EOS:3};
let vocab={charToId:{"<PAD>":0,"<UNK>":1,"<BOS>":2,"<EOS>":3},idToChar:["<PAD>","<UNK>","<BOS>","<EOS>"]};
function addChars(t){for(const c of t)if(!(c in vocab.charToId)){vocab.charToId[c]=vocab.idToChar.length;vocab.idToChar.push(c);}}
function encode(text){
  const out=new Int32Array(text.length+2);
  out[0]=SPECIAL.BOS;
  for(let i=0;i<text.length;i++)out[i+1]=vocab.charToId[text[i]]??SPECIAL.UNK;
  out[text.length+1]=SPECIAL.EOS;
  return out;
}
function decode(ids){return ids.filter(i=>i>SPECIAL.EOS).map(i=>vocab.idToChar[i]??"").join("");}

// ══════════════════════════════════════════════════════════════
// MATRIX
// ══════════════════════════════════════════════════════════════
function mat(r,c){return{data:new Float32Array(r*c),rows:r,cols:c};}
function matR(r,c,s){const m=mat(r,c);for(let i=0;i<m.data.length;i++)m.data[i]=(Math.random()*2-1)*s;return m;}
function f32B64(f){return Buffer.from(f.buffer).toString("base64");}
function b64F32(b){return new Float32Array(Buffer.from(b,"base64").buffer);}
function netToJSON(){
  if(!NET)return null;
  const e=m=>f32B64(m.data);
  return{vocab:{charToId:vocab.charToId,idToChar:vocab.idToChar},netCfg,trainSteps,enc:"b64f32",
    E:e(NET.E),W1:e(NET.W1),b1:e(NET.b1),W2:e(NET.W2),b2:e(NET.b2),
    mW1:e(NET.mW1),mb1:e(NET.mb1),mW2:e(NET.mW2),mb2:e(NET.mb2)};
}
function netFromJSON(d){
  if(d.net&&d.net.vocab)d=d.net;
  vocab={charToId:d.vocab.charToId,idToChar:d.vocab.idToChar};
  netCfg=d.netCfg||netCfg;trainSteps=d.trainSteps||0;
  buildNet();
  const dec=v=>typeof v==="string"?b64F32(v):new Float32Array(v);
  const ld=(m,s)=>{const a=dec(s);for(let i=0;i<m.data.length;i++)m.data[i]=a[i];};
  ld(NET.E,d.E);ld(NET.W1,d.W1);ld(NET.b1,d.b1);ld(NET.W2,d.W2);ld(NET.b2,d.b2);
  if(d.mW1)ld(NET.mW1,d.mW1);if(d.mb1)ld(NET.mb1,d.mb1);
  if(d.mW2)ld(NET.mW2,d.mW2);if(d.mb2)ld(NET.mb2,d.mb2);
}

// ══════════════════════════════════════════════════════════════
// NETWORK
// ══════════════════════════════════════════════════════════════
let NET=null,netCfg={hiddenSize:HIDDEN,ctxLen:CTX,embedDim:EMBED},trainSteps=0,recentLoss=null;
function buildNet(){
  const V=vocab.idToChar.length,{hiddenSize,ctxLen,embedDim}=netCfg,inDim=embedDim*ctxLen;
  NET={E:matR(V,embedDim,.05),dE:mat(V,embedDim),
    W1:matR(inDim,hiddenSize,Math.sqrt(2/inDim)),b1:mat(1,hiddenSize),dW1:mat(inDim,hiddenSize),db1:mat(1,hiddenSize),mW1:mat(inDim,hiddenSize),mb1:mat(1,hiddenSize),
    W2:matR(hiddenSize,V,Math.sqrt(2/hiddenSize)),b2:mat(1,V),dW2:mat(hiddenSize,V),db2:mat(1,V),mW2:mat(hiddenSize,V),mb2:mat(1,V),
    V,hiddenSize,ctxLen,embedDim,inDim};
}
function rebuildVocabExpand(){
  if(!NET){buildNet();return;}
  const oV=NET.V,nV=vocab.idToChar.length;if(nV<=oV)return;
  const{embedDim,hiddenSize}=NET;
  const nE=matR(nV,embedDim,.05);for(let i=0;i<oV;i++)for(let j=0;j<embedDim;j++)nE.data[i*embedDim+j]=NET.E.data[i*embedDim+j];
  NET.E=nE;NET.dE=mat(nV,embedDim);
  const nW2=matR(hiddenSize,nV,Math.sqrt(2/hiddenSize));for(let i=0;i<hiddenSize;i++)for(let j=0;j<oV;j++)nW2.data[i*nV+j]=NET.W2.data[i*oV+j];
  NET.W2=nW2;NET.dW2=mat(hiddenSize,nV);NET.mW2=mat(hiddenSize,nV);
  const nb=mat(1,nV);for(let j=0;j<oV;j++)nb.data[j]=NET.b2.data[j];
  NET.b2=nb;NET.db2=mat(1,nV);NET.mb2=mat(1,nV);NET.V=nV;
}
function forward(ctx){
  const{E,W1,b1,W2,b2,hiddenSize,ctxLen,embedDim,inDim,V}=NET;
  const ef=new Float32Array(inDim);
  for(let i=0;i<ctxLen;i++){const id=ctx[i]??0,base=id*embedDim,out=i*embedDim;for(let j=0;j<embedDim;j++)ef[out+j]=E.data[base+j];}
  const hp=new Float32Array(hiddenSize);
  for(let j=0;j<hiddenSize;j++){let s=b1.data[j];for(let i=0;i<inDim;i++)s+=ef[i]*W1.data[i*hiddenSize+j];hp[j]=s;}
  const h=new Float32Array(hiddenSize);for(let j=0;j<hiddenSize;j++)h[j]=hp[j]>0?hp[j]:0;
  const lo=new Float32Array(V);for(let k=0;k<V;k++){let s=b2.data[k];for(let j=0;j<hiddenSize;j++)s+=h[j]*W2.data[j*V+k];lo[k]=s;}
  return{embedFlat:ef,hPre:hp,h,logits:lo,ctx};
}
function softmax(lg,t=1){const tm=Math.max(t,.05);let mx=-Infinity;for(const v of lg)if(v>mx)mx=v;const e=new Float32Array(lg.length);let s=0;for(let i=0;i<lg.length;i++){e[i]=Math.exp((lg[i]-mx)/tm);s+=e[i];}for(let i=0;i<lg.length;i++)e[i]/=s;return e;}
function sampleP(p){let r=Math.random(),c=0;for(let i=0;i<p.length;i++){c+=p[i];if(r<c)return i;}return p.length-1;}
function backward(fwd,tid,lr){
  const{embedFlat:ef,hPre:hp,h,logits,ctx}=fwd;
  const{E,W1,b1,W2,b2,mW1,mb1,mW2,mb2,hiddenSize,ctxLen,embedDim,inDim,V}=NET;
  const MOM=.9,pr=softmax(logits,1),dL=pr.slice();
  dL[tid]-=1;const loss=-Math.log(Math.max(pr[tid],1e-9));
  for(let j=0;j<hiddenSize;j++)for(let k=0;k<V;k++){const g=h[j]*dL[k];mW2.data[j*V+k]=MOM*mW2.data[j*V+k]+(1-MOM)*g;W2.data[j*V+k]-=lr*mW2.data[j*V+k];}
  for(let k=0;k<V;k++){mb2.data[k]=MOM*mb2.data[k]+(1-MOM)*dL[k];b2.data[k]-=lr*mb2.data[k];}
  const dH=new Float32Array(hiddenSize);for(let j=0;j<hiddenSize;j++){let g=0;for(let k=0;k<V;k++)g+=W2.data[j*V+k]*dL[k];dH[j]=hp[j]>0?g:0;}
  for(let i=0;i<inDim;i++)for(let j=0;j<hiddenSize;j++){const g=ef[i]*dH[j];mW1.data[i*hiddenSize+j]=MOM*mW1.data[i*hiddenSize+j]+(1-MOM)*g;W1.data[i*hiddenSize+j]-=lr*mW1.data[i*hiddenSize+j];}
  for(let j=0;j<hiddenSize;j++){mb1.data[j]=MOM*mb1.data[j]+(1-MOM)*dH[j];b1.data[j]-=lr*mb1.data[j];}
  const de=new Float32Array(inDim);for(let i=0;i<inDim;i++){let g=0;for(let j=0;j<hiddenSize;j++)g+=W1.data[i*hiddenSize+j]*dH[j];de[i]=g;}
  for(let i=0;i<ctxLen;i++){const id=ctx[i]??0,base=id*embedDim,src=i*embedDim;for(let j=0;j<embedDim;j++)E.data[base+j]-=lr*de[src+j];}
  return loss;
}
function generate(seed="",maxLen=80,temp=.8){
  if(!NET)return"...";
  const{ctxLen}=NET;addChars(seed);rebuildVocabExpand();
  const si=encode(seed),ctx=new Array(ctxLen).fill(SPECIAL.PAD);
  for(let i=0;i<si.length;i++){const sl=si.slice(Math.max(0,i-ctxLen+1),i+1);ctx.fill(SPECIAL.PAD);for(let j=0;j<sl.length;j++)ctx[ctxLen-sl.length+j]=sl[j];}
  const out=[];
  for(let s=0;s<maxLen;s++){const fwd=forward(ctx),pr=softmax(fwd.logits,temp);pr[SPECIAL.PAD]=0;pr[SPECIAL.UNK]=0;pr[SPECIAL.BOS]=0;const sum=pr.reduce((a,b)=>a+b,0);if(sum>0)for(let i=0;i<pr.length;i++)pr[i]/=sum;const next=sampleP(pr);if(next===SPECIAL.EOS)break;out.push(next);ctx.copyWithin(0,1);ctx[ctxLen-1]=next;}
  return decode(out).trim()||"...";
}

// ══════════════════════════════════════════════════════════════
// PERSIST — model + training state (book index + char position)
// ══════════════════════════════════════════════════════════════
function saveModel(){
  const d=netToJSON();if(!d)return;
  const tmp=MODEL_FILE+".tmp";
  fs.writeFileSync(tmp,JSON.stringify(d),"utf8");
  fs.renameSync(tmp,MODEL_FILE);
}
function loadModel(){
  if(!fs.existsSync(MODEL_FILE))return false;
  try{netFromJSON(JSON.parse(fs.readFileSync(MODEL_FILE,"utf8")));return true;}
  catch(e){console.error(f("red","  load err: ")+e.message);return false;}
}

// Training state — which book and char position we're at
function loadState(){
  if(!fs.existsSync(STATE_FILE))return{bookIdx:0,charPos:0,startedAt:Date.now(),totalElapsedMs:0};
  try{return JSON.parse(fs.readFileSync(STATE_FILE,"utf8"));}
  catch{return{bookIdx:0,charPos:0,startedAt:Date.now(),totalElapsedMs:0};}
}
function saveState(state){
  fs.writeFileSync(STATE_FILE,JSON.stringify(state,null,2),"utf8");
}

// ══════════════════════════════════════════════════════════════
// BOOK DISCOVERY — reads books/ folder in sorted order
// ══════════════════════════════════════════════════════════════
function getBooks(){
  if(!fs.existsSync(BOOKS_DIR)){
    // fallback: check root for book.txt book1.txt etc
    const root=["book.txt","book1.txt","book2.txt","book3.txt","book4.txt","book5.txt"]
      .filter(f=>fs.existsSync(f));
    if(root.length)return root;
    console.error(f("red",`  ✗ No books found. Create a "books/" folder with .txt files.`));
    process.exit(1);
  }
  return fs.readdirSync(BOOKS_DIR)
    .filter(f=>f.endsWith(".txt"))
    .sort()
    .map(f=>path.join(BOOKS_DIR,f));
}

// ══════════════════════════════════════════════════════════════
// MAIN TRAINING LOOP
// ══════════════════════════════════════════════════════════════
async function main(){
  console.log();
  console.log(bf("magenta","  ╔════════════════════════════════════════════════════╗"));
  console.log(bf("magenta","  ║")+bf("white","    NeuroChat Multi-Book GitHub Trainer 🚀        ")+bf("magenta","║"));
  console.log(bf("magenta","  ╚════════════════════════════════════════════════════╝\n"));

  // Load model
  if(loadModel()){
    netCfg.hiddenSize=NET.hiddenSize;netCfg.ctxLen=NET.ctxLen;netCfg.embedDim=NET.embedDim;
    console.log(f("green",`  ✔ Resumed model — vocab:${vocab.idToChar.length} steps:${trainSteps.toLocaleString()}`));
  }else{
    netCfg={hiddenSize:HIDDEN,ctxLen:CTX,embedDim:EMBED};buildNet();
    console.log(f("yellow",`  ◆ Fresh model h:${HIDDEN} ctx:${CTX} embed:${EMBED}`));
  }

  const books=getBooks();
  const state=loadState();
  const runStart=Date.now();

  // Total elapsed across ALL runs (persisted in state)
  const prevElapsed=state.totalElapsedMs||0;

  console.log(f("cyan",`  Books     : `)+books.map((b,i)=>`[${i+1}] ${path.basename(b)}`).join("  "));
  console.log(f("cyan",`  Target    : `)+TARGET_DAYS+" days");
  console.log(f("cyan",`  This run  : `)+MAX_HOURS+"h  (exit 2 → restart)");
  console.log(f("cyan",`  Save      : `)+SAVE_MS/1000+"s");
  console.log(f("cyan",`  Resuming  : `)+`book[${state.bookIdx+1}/${books.length}] charPos:${state.charPos.toLocaleString()}`);
  console.log(f("cyan",`  Progress  : `)+`${(prevElapsed/TARGET_MS*100).toFixed(1)}% of target\n`);

  // Check if already finished
  if(fs.existsSync("finished")){
    console.log(bf("green","  🎉 Already finished! Delete 'finished' file to retrain."));
    process.exit(0);
  }

  // ── Graceful shutdown ──────────────────────────────────────
  let shuttingDown=false;
  function gracefulExit(code){
    if(shuttingDown)return;shuttingDown=true;
    cl();console.log(f("yellow",`\n  ⚡ saving before exit...`));
    state.totalElapsedMs=prevElapsed+(Date.now()-runStart);
    saveState(state);saveModel();
    console.log(f("green",`  💾 saved — steps:${trainSteps.toLocaleString()}`));
    process.exit(code);
  }
  process.on("SIGTERM",()=>gracefulExit(2));
  process.on("SIGINT", ()=>gracefulExit(0));

  // ── Load current book (chunked addChars for 50MB safety) ──
  let bookIdx  = state.bookIdx % books.length;
  let charPos  = state.charPos;
  let ids      = null;
  let bookText = null;

  async function loadBook(idx){
    const bpath=books[idx];
    console.log(f("blue",`\n  📖 Loading book [${idx+1}/${books.length}]: ${path.basename(bpath)}`));
    bookText=fs.readFileSync(bpath,"utf8");
    console.log(f("dim",`     ${(bookText.length/1024/1024).toFixed(2)}MB`));
    // stream addChars
    const PREP=256*1024;
    for(let i=0;i<bookText.length;i+=PREP){
      addChars(bookText.slice(i,i+PREP));
      if(i%(PREP*8)===0)await new Promise(r=>setImmediate(r));
    }
    rebuildVocabExpand();
    cl();process.stdout.write(f("yellow","  ⏳ encoding..."));
    await new Promise(r=>setImmediate(r));
    ids=encode(bookText);
    cl();console.log(f("green",`  ✔ ${ids.length.toLocaleString()} tokens  vocab:${vocab.idToChar.length}`));
  }

  await loadBook(bookIdx);

  // ── Training tick ──────────────────────────────────────────
  const ctx=new Array(CTX).fill(0);
  let lossAcc=0,lossCnt=0;
  let lastSave=Date.now(),lastViz=Date.now(),lastStatus=Date.now();
  const lossHist=[];
  let speedWin=[],totalThis=0;

  return new Promise(resolve=>{
    async function tick(){
      if(shuttingDown)return;
      const now=Date.now();
      const runElapsed=now-runStart;
      const totalElapsed=prevElapsed+runElapsed;

      // ── Time limit for this run ──────────────────────────
      if(runElapsed>=MAX_MS){
        cl();console.log(bf("yellow","\n  ⏰ 4h limit — restarting..."));
        state.bookIdx=bookIdx;state.charPos=charPos;
        state.totalElapsedMs=totalElapsed;
        saveState(state);saveModel();
        console.log(f("green","  💾 saved"));
        process.exit(2);
      }

      // ── Check if total target days reached ───────────────
      if(totalElapsed>=TARGET_MS){
        cl();console.log();
        console.log(bf("green",`  🎉 TARGET REACHED — ${TARGET_DAYS} days of training complete!`));
        console.log(bf("green",`  📊 Total steps: ${trainSteps.toLocaleString()}`));
        state.totalElapsedMs=totalElapsed;
        saveState(state);saveModel();
        // Create finished file
        fs.writeFileSync("finished",JSON.stringify({
          completedAt:new Date().toISOString(),
          totalSteps:trainSteps,
          totalDays:TARGET_DAYS,
          vocab:vocab.idToChar.length,
          hidden:HIDDEN,
          books:books.map(b=>path.basename(b)),
        },null,2),"utf8");
        console.log(f("cyan","  📄 created 'finished' file"));
        process.exit(0);
      }

      // ── Training frame ───────────────────────────────────
      const deadline=now+FRAME_MS;
      let stepsThis=0;
      while(Date.now()<deadline){
        // Book finished → move to next book
        if(charPos+1>=ids.length){
          charPos=0;bookIdx=(bookIdx+1)%books.length;
          cl();console.log(f("blue",`\n  📖 switching to book [${bookIdx+1}/${books.length}]: ${path.basename(books[bookIdx])}`));
          state.bookIdx=bookIdx;state.charPos=0;
          // Load next book async — break out of frame, reload, resume
          saveState(state);saveModel();
          loadBook(bookIdx).then(()=>setImmediate(tick));
          return;
        }
        const sg=Math.max(0,charPos-CTX+1),sl=charPos+1-sg;
        ctx.fill(0);for(let j=0;j<sl;j++)ctx[CTX-sl+j]=ids[sg+j];
        lossAcc+=backward(forward(ctx),ids[charPos+1],LR);
        lossCnt++;charPos++;trainSteps++;stepsThis++;totalThis++;
      }
      if(lossCnt>0)recentLoss=lossAcc/lossCnt;

      // rolling speed
      speedWin.push({t:now,s:totalThis});
      speedWin=speedWin.filter(x=>now-x.t<5000);
      const sps=speedWin.length>1?Math.round((speedWin[speedWin.length-1].s-speedWin[0].s)/((speedWin[speedWin.length-1].t-speedWin[0].t)/1000)):0;

      // 60s save
      if(now-lastSave>=SAVE_MS){
        lastSave=now;
        cl();process.stdout.write(f("blue","  💾 saving..."));
        state.bookIdx=bookIdx;state.charPos=charPos;
        state.totalElapsedMs=totalElapsed;
        saveState(state);saveModel();
        lossHist.push(recentLoss??0);if(lossHist.length>60)lossHist.shift();
        lossAcc=0;lossCnt=0;
      }

      // viz
      if(now-lastViz>=VIZ_MS){
        lastViz=now;
        console.log("\n"+"─".repeat(60));
        drawViz(NET,lossHist,sps,runElapsed,path.basename(books[bookIdx]),bookIdx,books.length);
        const sample=generate(bookText?.trim().split(/\s+/).slice(0,2).join(" ")||"",80,.85);
        console.log(f("yellow","  sample → ")+f("cyan",sample));
        console.log("─".repeat(60));
      }

      // status line
      if(now-lastStatus>=500){
        lastStatus=now;
        const pct=totalElapsed/TARGET_MS;
        const bpct=(charPos/ids.length*100).toFixed(1);
        cl();process.stdout.write(
          `  ${bar(Math.min(pct,1),24)}`+
          f("dim",` ${(pct*100).toFixed(1)}% of ${TARGET_DAYS}d`)+
          f("cyan",`  ${sps.toLocaleString()}sps`)+
          f("green",`  loss:${recentLoss!=null?recentLoss.toFixed(4):"—"}`)+
          f("blue",`  book${bookIdx+1}:${bpct}%`)+
          f("dim",`  steps:${trainSteps.toLocaleString()}`)
        );
      }

      setImmediate(tick);
    }
    setImmediate(tick);
  });
}

main().catch(e=>{console.error(f("red","  ✗ ")+e.message+"\n"+e.stack);process.exit(1);});
