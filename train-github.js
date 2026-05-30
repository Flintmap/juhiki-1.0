#!/usr/bin/env node
// NeuroChat GitHub Trainer 2.0
// - Typed arrays + reused buffers (faster, less GC)
// - Atomic save (no mixed top/bottom halves)
// - Auto-detects model.json
// - Multi-book with loop count per book
// - Neuron heatmap, steps/sec, loss sparkline
"use strict";
const fs = require("fs");
const path = require("path");

// ── CLI ───────────────────────────────────────────────────────
const args = process.argv.slice(2);
const flags = {};
const pos = [];
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith("--")) { flags[args[i].slice(2)] = args[i+1]; i++; }
  else pos.push(args[i]);
}
const BOOKS_DIR  = flags.books  || pos[0] || "books";
const MODEL_FILE = flags.model  || "model.json";
const STATE_FILE = flags.state  || "train-state.json";
const LR         = parseFloat(flags.lr    || "0.01");
const HIDDEN     = parseInt(flags.hidden  || "1024", 10);
const CTX        = parseInt(flags.ctx     || "128",  10);
const EMBED      = parseInt(flags.embed   || "64",  10);
const LOOPS      = parseInt(flags.loops   || "20",  10);
const SAVE_MS    = parseInt(flags.save    || "60",  10) * 1000;
const MAX_HOURS  = parseFloat(flags.hours || "4");
const MAX_MS     = MAX_HOURS * 3600000;
const FRAME_MS   = parseInt(flags.framems || "80",  10);
const VIZ_MS     = parseInt(flags.viz     || "300", 10) * 1000;

// ── Colours ───────────────────────────────────────────────────
const C = {
  reset:"\x1b[0m",bold:"\x1b[1m",dim:"\x1b[2m",
  cyan:"\x1b[36m",green:"\x1b[32m",yellow:"\x1b[33m",
  red:"\x1b[31m",magenta:"\x1b[35m",blue:"\x1b[34m",white:"\x1b[37m",
};
const f  = (c,t) => `${C[c]}${t}${C.reset}`;
const bf = (c,t) => `${C.bold}${C[c]}${t}${C.reset}`;
function bar(p,w=30){const n=Math.min(w,Math.round(p*w));return f("cyan","█".repeat(n))+f("dim","░".repeat(w-n));}
function cl(){process.stdout.write("\r\x1b[K");}

// ── Neuron viz ────────────────────────────────────────────────
const HC=["\x1b[38;5;17m","\x1b[38;5;19m","\x1b[38;5;21m","\x1b[38;5;33m","\x1b[38;5;45m","\x1b[38;5;51m","\x1b[38;5;82m","\x1b[38;5;118m","\x1b[38;5;226m","\x1b[38;5;196m"];
const HH=[" ","·","▪","▫","◆","◈","●","◉","⬟","■"];
function hc(v,mn,mx){if(mx===mn)return HC[0]+HH[0]+C.reset;const n=Math.max(0,Math.min(1,(Math.abs(v)-mn)/(mx-mn)));const i=Math.min(9,Math.floor(n*10));return HC[i]+HH[i]+C.reset;}

function drawViz(lossHist,sps,elapsed,bookName,loop,totalLoops){
  if(!NET)return;
  const{W1,b1,W2,hiddenSize,embedDim,ctxLen,V}=NET;
  console.log();
  console.log(bf("magenta","  ╔════════════════════════════════════════════════════╗"));
  console.log(bf("magenta","  ║")+bf("white","           🧠  NEURON ACTIVITY MAP                 ")+bf("magenta","║"));
  console.log(bf("magenta","  ╚════════════════════════════════════════════════════╝"));
  const r1=Math.min(14,hiddenSize),c1=Math.min(40,embedDim*ctxLen);
  let mn=Infinity,mx=-Infinity;
  for(let r=0;r<r1;r++)for(let c=0;c<c1;c++){const v=W1.data[r*W1.cols+c];if(v<mn)mn=v;if(v>mx)mx=v;}
  console.log(f("cyan",`\n  W1 [${r1}×${c1}]`));
  for(let r=0;r<r1;r++){process.stdout.write("  ");for(let c=0;c<c1;c++)process.stdout.write(hc(W1.data[r*W1.cols+c],Math.abs(mn),Math.abs(mx)));process.stdout.write("\n");}
  const r2=Math.min(14,hiddenSize),c2=Math.min(40,V);
  let mn2=Infinity,mx2=-Infinity;
  for(let r=0;r<r2;r++)for(let c=0;c<c2;c++){const v=W2.data[r*W2.cols+c];if(v<mn2)mn2=v;if(v>mx2)mx2=v;}
  console.log(f("green",`\n  W2 [${r2}×${c2}]`));
  for(let r=0;r<r2;r++){process.stdout.write("  ");for(let c=0;c<c2;c++)process.stdout.write(hc(W2.data[r*W2.cols+c],Math.abs(mn2),Math.abs(mx2)));process.stdout.write("\n");}
  console.log(f("yellow",`\n  b1 [${hiddenSize}]`));
  process.stdout.write("  ");
  let bMn=Infinity,bMx=-Infinity;
  for(const v of b1.data){if(v<bMn)bMn=v;if(v>bMx)bMx=v;}
  for(let i=0;i<Math.min(60,hiddenSize);i++)process.stdout.write(hc(b1.data[i],Math.abs(bMn),Math.abs(bMx)));
  process.stdout.write("\n");
  if(lossHist.length>1){
    const lMx=Math.max(...lossHist),lMn=Math.min(...lossHist);
    const sp="▁▂▃▄▅▆▇█";
    console.log(f("magenta",`\n  loss [${lossHist.length} saves]`));
    process.stdout.write("  ");
    for(const v of lossHist){const n=lMx>lMn?(v-lMn)/(lMx-lMn):0;const i=Math.min(7,Math.floor(n*8));const col=n>0.7?C.red:n>0.4?C.yellow:C.green;process.stdout.write(col+sp[i]+C.reset);}
    process.stdout.write(f("dim",`  ${lMn.toFixed(4)}→${lMx.toFixed(4)}\n`));
  }
  const em=(elapsed/60000).toFixed(1),rm=((MAX_MS-elapsed)/60000).toFixed(1);
  console.log();
  console.log(f("cyan","  steps/s : ")+bf("white",sps.toLocaleString())+f("cyan","   steps : ")+bf("white",trainSteps.toLocaleString())+f("cyan","   vocab : ")+bf("white",String(vocab.idToChar.length)));
  console.log(f("green","  elapsed : ")+bf("white",em+"min")+f("green","   left : ")+bf("white",rm+"min")+f("green","   loss : ")+bf("yellow",recentLoss!=null?recentLoss.toFixed(5):"—"));
  console.log(f("blue","  book    : ")+bf("white",bookName)+f("blue","   loop : ")+bf("white",`${loop}/${totalLoops}`));
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
// NETWORK — reusable buffers allocated once
// ══════════════════════════════════════════════════════════════
let NET=null,netCfg={hiddenSize:HIDDEN,ctxLen:CTX,embedDim:EMBED};
let trainSteps=0,recentLoss=null;
let _ef=null,_hp=null,_h=null,_lo=null,_dL=null,_dH=null,_de=null;

function buildNet(){
  const V=vocab.idToChar.length,{hiddenSize,ctxLen,embedDim}=netCfg,inDim=embedDim*ctxLen;
  NET={
    E:matR(V,embedDim,.05),dE:mat(V,embedDim),
    W1:matR(inDim,hiddenSize,Math.sqrt(2/inDim)),b1:mat(1,hiddenSize),
    dW1:mat(inDim,hiddenSize),db1:mat(1,hiddenSize),mW1:mat(inDim,hiddenSize),mb1:mat(1,hiddenSize),
    W2:matR(hiddenSize,V,Math.sqrt(2/hiddenSize)),b2:mat(1,V),
    dW2:mat(hiddenSize,V),db2:mat(1,V),mW2:mat(hiddenSize,V),mb2:mat(1,V),
    V,hiddenSize,ctxLen,embedDim,inDim,
  };
  _ef=new Float32Array(inDim);_hp=new Float32Array(hiddenSize);
  _h=new Float32Array(hiddenSize);_lo=new Float32Array(V);
  _dL=new Float32Array(V);_dH=new Float32Array(hiddenSize);_de=new Float32Array(inDim);
}

function rebuildVocabExpand(){
  if(!NET){buildNet();return;}
  const oV=NET.V,nV=vocab.idToChar.length;if(nV<=oV)return;
  const{embedDim,hiddenSize}=NET;
  const nE=matR(nV,embedDim,.05);
  for(let i=0;i<oV;i++)for(let j=0;j<embedDim;j++)nE.data[i*embedDim+j]=NET.E.data[i*embedDim+j];
  NET.E=nE;NET.dE=mat(nV,embedDim);
  const nW2=matR(hiddenSize,nV,Math.sqrt(2/hiddenSize));
  for(let i=0;i<hiddenSize;i++)for(let j=0;j<oV;j++)nW2.data[i*nV+j]=NET.W2.data[i*oV+j];
  NET.W2=nW2;NET.dW2=mat(hiddenSize,nV);NET.mW2=mat(hiddenSize,nV);
  const nb=mat(1,nV);for(let j=0;j<oV;j++)nb.data[j]=NET.b2.data[j];
  NET.b2=nb;NET.db2=mat(1,nV);NET.mb2=mat(1,nV);NET.V=nV;
  // reallocate buffers for new V
  _lo=new Float32Array(nV);_dL=new Float32Array(nV);
}

function forward(ctx){
  const{E,W1,b1,W2,b2,hiddenSize,ctxLen,embedDim,inDim,V}=NET;
  _ef.fill(0);
  for(let i=0;i<ctxLen;i++){const id=ctx[i]??0,base=id*embedDim,out=i*embedDim;for(let j=0;j<embedDim;j++)_ef[out+j]=E.data[base+j];}
  for(let j=0;j<hiddenSize;j++){let s=b1.data[j];for(let i=0;i<inDim;i++)s+=_ef[i]*W1.data[i*hiddenSize+j];_hp[j]=s;}
  for(let j=0;j<hiddenSize;j++)_h[j]=_hp[j]>0?_hp[j]:0;
  for(let k=0;k<V;k++){let s=b2.data[k];for(let j=0;j<hiddenSize;j++)s+=_h[j]*W2.data[j*V+k];_lo[k]=s;}
  return{ef:_ef,hp:_hp,h:_h,lo:_lo,ctx};
}

function softmaxBuf(buf,temp=1.0){
  const t=Math.max(temp,.05);let mx=-Infinity;
  for(let i=0;i<buf.length;i++)if(buf[i]>mx)mx=buf[i];
  let s=0;for(let i=0;i<buf.length;i++){buf[i]=Math.exp((buf[i]-mx)/t);s+=buf[i];}
  for(let i=0;i<buf.length;i++)buf[i]/=s;
}
function sampleP(p){let r=Math.random(),c=0;for(let i=0;i<p.length;i++){c+=p[i];if(r<c)return i;}return p.length-1;}

function backward(fwd,tid,lr){
  const{ef,hp,h,lo,ctx}=fwd;
  const{E,W1,b1,W2,b2,mW1,mb1,mW2,mb2,hiddenSize,ctxLen,embedDim,inDim,V}=NET;
  const MOM=.9;
  for(let i=0;i<V;i++)_dL[i]=lo[i];
  softmaxBuf(_dL,1.0);
  const loss=-Math.log(Math.max(_dL[tid],1e-9));
  _dL[tid]-=1.0;
  for(let j=0;j<hiddenSize;j++)for(let k=0;k<V;k++){const g=h[j]*_dL[k];mW2.data[j*V+k]=MOM*mW2.data[j*V+k]+(1-MOM)*g;W2.data[j*V+k]-=lr*mW2.data[j*V+k];}
  for(let k=0;k<V;k++){mb2.data[k]=MOM*mb2.data[k]+(1-MOM)*_dL[k];b2.data[k]-=lr*mb2.data[k];}
  for(let j=0;j<hiddenSize;j++){let g=0;for(let k=0;k<V;k++)g+=W2.data[j*V+k]*_dL[k];_dH[j]=hp[j]>0?g:0;}
  for(let i=0;i<inDim;i++)for(let j=0;j<hiddenSize;j++){const g=ef[i]*_dH[j];mW1.data[i*hiddenSize+j]=MOM*mW1.data[i*hiddenSize+j]+(1-MOM)*g;W1.data[i*hiddenSize+j]-=lr*mW1.data[i*hiddenSize+j];}
  for(let j=0;j<hiddenSize;j++){mb1.data[j]=MOM*mb1.data[j]+(1-MOM)*_dH[j];b1.data[j]-=lr*mb1.data[j];}
  for(let i=0;i<inDim;i++){let g=0;for(let j=0;j<hiddenSize;j++)g+=W1.data[i*hiddenSize+j]*_dH[j];_de[i]=g;}
  for(let i=0;i<ctxLen;i++){const id=ctx[i]??0,base=id*embedDim,src=i*embedDim;for(let j=0;j<embedDim;j++)E.data[base+j]-=lr*_de[src+j];}
  return loss;
}

function generate(seed="",maxLen=80,temp=.8){
  if(!NET)return"...";
  const{ctxLen,inDim,hiddenSize,V,embedDim,E,W1,b1,W2,b2}=NET;
  addChars(seed);rebuildVocabExpand();
  const si=encode(seed),ctx=new Int32Array(ctxLen);
  for(let i=0;i<si.length;i++){const sl=si.slice(Math.max(0,i-ctxLen+1),i+1);ctx.fill(0);for(let j=0;j<sl.length;j++)ctx[ctxLen-sl.length+j]=sl[j];}
  const out=[];
  const gEf=new Float32Array(inDim),gHp=new Float32Array(hiddenSize),gH=new Float32Array(hiddenSize),gLo=new Float32Array(V);
  for(let s=0;s<maxLen;s++){
    gEf.fill(0);for(let i=0;i<ctxLen;i++){const id=ctx[i]??0,base=id*embedDim,o=i*embedDim;for(let j=0;j<embedDim;j++)gEf[o+j]=E.data[base+j];}
    for(let j=0;j<hiddenSize;j++){let sum=b1.data[j];for(let i=0;i<inDim;i++)sum+=gEf[i]*W1.data[i*hiddenSize+j];gHp[j]=sum;}
    for(let j=0;j<hiddenSize;j++)gH[j]=gHp[j]>0?gHp[j]:0;
    for(let k=0;k<V;k++){let sum=b2.data[k];for(let j=0;j<hiddenSize;j++)sum+=gH[j]*W2.data[j*V+k];gLo[k]=sum;}
    softmaxBuf(gLo,temp);
    gLo[SPECIAL.PAD]=0;gLo[SPECIAL.UNK]=0;gLo[SPECIAL.BOS]=0;
    const sum=gLo.reduce((a,b)=>a+b,0);if(sum>0)for(let i=0;i<V;i++)gLo[i]/=sum;
    const next=sampleP(gLo);if(next===SPECIAL.EOS)break;
    out.push(next);ctx.copyWithin(0,1);ctx[ctxLen-1]=next;
  }
  return decode(out).trim()||"...";
}

// ══════════════════════════════════════════════════════════════
// ATOMIC SAVE
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
function loadState(){
  if(!fs.existsSync(STATE_FILE))return{bookIdx:0,charPos:0,loop:0,totalElapsedMs:0};
  try{return JSON.parse(fs.readFileSync(STATE_FILE,"utf8"));}
  catch{return{bookIdx:0,charPos:0,loop:0,totalElapsedMs:0};}
}
function saveState(s){
  const tmp=STATE_FILE+".tmp";
  fs.writeFileSync(tmp,JSON.stringify(s,null,2),"utf8");
  fs.renameSync(tmp,STATE_FILE);
}

// ══════════════════════════════════════════════════════════════
// BOOKS
// ══════════════════════════════════════════════════════════════
function getBooks(){
  if(fs.existsSync(BOOKS_DIR)&&fs.statSync(BOOKS_DIR).isDirectory()){
    const list=fs.readdirSync(BOOKS_DIR).filter(f=>f.endsWith(".txt")).sort().map(f=>path.join(BOOKS_DIR,f));
    if(list.length)return list;
  }
  if(fs.existsSync(BOOKS_DIR))return[BOOKS_DIR];
  if(fs.existsSync("book.txt"))return["book.txt"];
  console.error(f("red","  ✗ no books found"));process.exit(1);
}

// ══════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════
async function main(){
  console.log();
  console.log(bf("magenta","  ╔══════════════════════════════════════════════════╗"));
  console.log(bf("magenta","  ║")+bf("white","    NeuroChat GitHub Trainer 2.0 🚀             ")+bf("magenta","║"));
  console.log(bf("magenta","  ╚══════════════════════════════════════════════════╝\n"));

  if(loadModel()){
    netCfg={hiddenSize:NET.hiddenSize,ctxLen:NET.ctxLen,embedDim:NET.embedDim};
    console.log(f("green",`  ✔ Loaded model — vocab:${vocab.idToChar.length} steps:${trainSteps.toLocaleString()}`));
  }else{
    netCfg={hiddenSize:HIDDEN,ctxLen:CTX,embedDim:EMBED};
    buildNet();
    console.log(f("yellow",`  ◆ Fresh model h:${HIDDEN} ctx:${CTX} embed:${EMBED}`));
  }

  const books=getBooks();
  const state=loadState();
  const runStart=Date.now();
  const prevElapsed=state.totalElapsedMs||0;

  console.log(f("cyan",`  Books  : `)+books.map((b,i)=>`[${i+1}]${path.basename(b)}`).join(" "));
  console.log(f("cyan",`  Loops  : `)+LOOPS+" per book");
  console.log(f("cyan",`  LR     : `)+LR+"  FRAME_MS:"+FRAME_MS);
  console.log(f("cyan",`  Save   : `)+SAVE_MS/1000+"s (atomic)");
  console.log(f("cyan",`  Resume : `)+`book[${state.bookIdx+1}] loop[${(state.loop||0)+1}/${LOOPS}] pos:${(state.charPos||0).toLocaleString()}\n`);

  let exiting=false;
  function gracefulExit(code){
    if(exiting)return;exiting=true;
    cl();console.log(f("yellow",`\n  ⚡ saving...`));
    state.totalElapsedMs=prevElapsed+(Date.now()-runStart);
    saveState(state);saveModel();
    console.log(f("green",`  💾 saved — steps:${trainSteps.toLocaleString()}`));
    process.exit(code);
  }
  process.on("SIGTERM",()=>gracefulExit(2));
  process.on("SIGINT", ()=>gracefulExit(0));

  let bookIdx=(state.bookIdx||0)%books.length;
  let charPos=state.charPos||0;
  let loop=state.loop||0;
  let ids=null,bookText=null;

  async function loadBook(idx){
    const bp=books[idx];
    console.log(f("blue",`\n  📖 [${idx+1}/${books.length}] ${path.basename(bp)}`));
    bookText=fs.readFileSync(bp,"utf8");
    const PREP=256*1024;
    for(let i=0;i<bookText.length;i+=PREP){
      addChars(bookText.slice(i,i+PREP));
      if(i%(PREP*8)===0)await new Promise(r=>setImmediate(r));
    }
    rebuildVocabExpand();
    cl();process.stdout.write(f("yellow","  ⏳ encoding..."));
    await new Promise(r=>setImmediate(r));
    ids=encode(bookText);
    cl();console.log(f("green",`  ✔ ${ids.length.toLocaleString()} tokens vocab:${vocab.idToChar.length}`));
  }

  await loadBook(bookIdx);

  const ctx=new Int32Array(CTX);
  let lossAcc=0,lossCnt=0;
  let lastSave=Date.now(),lastViz=Date.now(),lastStatus=Date.now();
  const lossHist=[];
  let speedWin=[],totalThis=0;

  return new Promise(resolve=>{
    async function tick(){
      if(exiting)return;
      const now=Date.now();
      const runElapsed=now-runStart;

      if(runElapsed>=MAX_MS){
        cl();console.log(bf("yellow","\n  ⏰ time limit — saving..."));
        state.bookIdx=bookIdx;state.charPos=charPos;state.loop=loop;
        state.totalElapsedMs=prevElapsed+runElapsed;
        saveState(state);saveModel();
        process.exit(2);
      }

      const deadline=now+FRAME_MS;
      let stepsThis=0;
      while(Date.now()<deadline){
        const N=ids.length;
        if(charPos+1>=N){
          charPos=0;loop++;
          if(loop>=LOOPS){
            loop=0;bookIdx=(bookIdx+1)%books.length;
            state.bookIdx=bookIdx;state.charPos=0;state.loop=0;
            saveState(state);saveModel();
            cl();console.log(f("blue",`\n  📖 next book [${bookIdx+1}/${books.length}]`));
            loadBook(bookIdx).then(()=>setImmediate(tick));
            return;
          }
          cl();console.log(f("dim",`  ↩ loop ${loop+1}/${LOOPS}`));
        }
        const sg=Math.max(0,charPos-CTX+1),sl=charPos+1-sg;
        ctx.fill(0);for(let j=0;j<sl;j++)ctx[CTX-sl+j]=ids[sg+j];
        lossAcc+=backward(forward(ctx),ids[charPos+1],LR);
        lossCnt++;charPos++;trainSteps++;stepsThis++;totalThis++;
      }
      if(lossCnt>0)recentLoss=lossAcc/lossCnt;

      speedWin.push({t:now,s:totalThis});
      speedWin=speedWin.filter(x=>now-x.t<5000);
      const sps=speedWin.length>1?Math.round((speedWin[speedWin.length-1].s-speedWin[0].s)/((speedWin[speedWin.length-1].t-speedWin[0].t)/1000)):0;

      if(now-lastSave>=SAVE_MS){
        lastSave=now;
        cl();process.stdout.write(f("blue","  💾 saving..."));
        state.bookIdx=bookIdx;state.charPos=charPos;state.loop=loop;
        state.totalElapsedMs=prevElapsed+runElapsed;
        saveState(state);saveModel();
        lossHist.push(recentLoss??0);if(lossHist.length>60)lossHist.shift();
        lossAcc=0;lossCnt=0;cl();
      }

      if(now-lastViz>=VIZ_MS){
        lastViz=now;
        console.log("\n"+"─".repeat(58));
        drawViz(lossHist,sps,runElapsed,path.basename(books[bookIdx]),loop+1,LOOPS);
        const sample=generate((bookText||"").trim().split(/\s+/).slice(0,2).join(" ")||"",60,.85);
        console.log(f("yellow","  sample → ")+f("cyan",sample));
        console.log("─".repeat(58));
      }

      if(now-lastStatus>=400){
        lastStatus=now;
        const N=ids.length,bpct=(charPos/N*100).toFixed(1);
        cl();process.stdout.write(
          `  ${bar(loop/LOOPS,20)}`+
          f("dim",` L${loop+1}/${LOOPS}`)+
          f("cyan",`  ${sps.toLocaleString()}sps`)+
          f("green",`  loss:${recentLoss!=null?recentLoss.toFixed(4):"—"}`)+
          f("blue",`  bk${bookIdx+1}:${bpct}%`)+
          f("dim",`  ${trainSteps.toLocaleString()}steps`)
        );
      }
      setImmediate(tick);
    }
    setImmediate(tick);
  });
}

main().catch(e=>{console.error(f("red","  ✗ ")+e.message);process.exit(1);});
