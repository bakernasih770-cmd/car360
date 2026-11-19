/* Content-based stitching: every photo finds its place by matching what is in it with its neighbours.
   Input:  {images:[Blob], qs:[[x,y,z,w]|null], hfov:number}
   Output: {qs:[[x,y,z,w]], hfov, matched:[bool], pairs:n}
   Camera convention: looks along -z, y up, x right (same as the viewer). */
'use strict';
let cv=null;
async function loadCV(){
  if(cv)return cv;
  const buf=await (await fetch('opencv_js.wasm')).arrayBuffer();
  await WebAssembly.compile(buf);
  self.Module={wasmBinary:buf};importScripts('opencv_nowasm.js');
  let c=self.cv;
  // the OpenCV module is itself "then-able": unwrap it by hand, otherwise await would wait for ever
  if(c&&!c.Mat)c=await new Promise(res=>{if(typeof c.then==='function')c.then(m=>{const mm=m;try{delete mm.then;}catch(e){}res(mm);});else c.onRuntimeInitialized=()=>res(c);});
  return cv=c;
}
const prog=(stage,i,n)=>postMessage({progress:{stage,i,n}});
// tolerances (pixels at 800 px width / degrees). Hand-held turning moves the lens a little, so near things shift: allow for it.
let TOL={ransac:6,pre:14,strict:6,agree:3,tilt:3};

/* ---- tiny quaternion helpers [x,y,z,w] ---- */
const qmul=(a,b)=>[a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]];
const qinv=a=>[-a[0],-a[1],-a[2],a[3]];
const qnorm=a=>{const n=Math.hypot(a[0],a[1],a[2],a[3])||1;return[a[0]/n,a[1]/n,a[2]/n,a[3]/n];};
const qdot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+a[3]*b[3];
const qang=(a,b)=>2*Math.acos(Math.min(1,Math.abs(qdot(a,b))));
function qrot(q,v){const p=qmul(qmul(q,[v[0],v[1],v[2],0]),qinv(q));return[p[0],p[1],p[2]];}
function qavg(list){ // weighted average of nearby quaternions
  const ref=list[0][0];let s=[0,0,0,0];
  for(const [q,w] of list){const sg=qdot(q,ref)<0?-1:1;for(let k=0;k<4;k++)s[k]+=sg*w*q[k];}
  return qnorm(s);}

/* exact eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (Jacobi rotations) */
function topEigen4(M){const a=M.map(r=>r.slice()),V=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  for(let sweep=0;sweep<50;sweep++){let off=0;for(let p=0;p<4;p++)for(let q=p+1;q<4;q++)off+=a[p][q]*a[p][q];if(off<1e-20)break;
    for(let p=0;p<4;p++)for(let q=p+1;q<4;q++){if(Math.abs(a[p][q])<1e-30)continue;
      const th=(a[q][q]-a[p][p])/(2*a[p][q]),t=Math.sign(th||1)/(Math.abs(th)+Math.sqrt(th*th+1)),c=1/Math.sqrt(t*t+1),sn=t*c;
      for(let k=0;k<4;k++){const akp=a[k][p],akq=a[k][q];a[k][p]=c*akp-sn*akq;a[k][q]=sn*akp+c*akq;}
      for(let k=0;k<4;k++){const apk=a[p][k],aqk=a[q][k];a[p][k]=c*apk-sn*aqk;a[q][k]=sn*apk+c*aqk;}
      for(let k=0;k<4;k++){const vkp=V[k][p],vkq=V[k][q];V[k][p]=c*vkp-sn*vkq;V[k][q]=sn*vkp+c*vkq;}}}
  let bi=0;for(let i=1;i<4;i++)if(a[i][i]>a[bi][bi])bi=i;return[V[0][bi],V[1][bi],V[2][bi],V[3][bi]];}
/* Horn: best rotation R with b ≈ R a for ray pairs (unit vectors) */
function horn(A,B){
  let Sxx=0,Sxy=0,Sxz=0,Syx=0,Syy=0,Syz=0,Szx=0,Szy=0,Szz=0;
  for(let i=0;i<A.length;i++){const a=A[i],b=B[i];Sxx+=a[0]*b[0];Sxy+=a[0]*b[1];Sxz+=a[0]*b[2];Syx+=a[1]*b[0];Syy+=a[1]*b[1];Syz+=a[1]*b[2];Szx+=a[2]*b[0];Szy+=a[2]*b[1];Szz+=a[2]*b[2];}
  const N=[[Sxx+Syy+Szz,Syz-Szy,Szx-Sxz,Sxy-Syx],[Syz-Szy,Sxx-Syy-Szz,Sxy+Syx,Szx+Sxz],[Szx-Sxz,Sxy+Syx,-Sxx+Syy-Szz,Syz+Szy],[Sxy-Syx,Szx+Sxz,Syz+Szy,-Sxx-Syy+Szz]];
  const v=topEigen4(N);return qnorm([v[1],v[2],v[3],v[0]]); // [x,y,z,w]
}
const fOf=(F,fov)=>(Math.max(F.W,F.H)/2)/Math.tan(fov*Math.PI/360); // focal length in pixels, upright or sideways
const ray=(x,y,W,H,f)=>{const v=[(x-W/2)/f,-(y-H/2)/f,-1],n=Math.hypot(v[0],v[1],v[2]);return[v[0]/n,v[1]/n,v[2]/n];};

/* ---- main ---- */
async function stitch({images,qs,hfov,fixFov,tol,fullSolve}){
  if(tol)TOL=Object.assign({},TOL,tol);
  const c=await loadCV();const n=images.length,FW=800,feats=[];
  const orb=new c.ORB(2500);
  for(let i=0;i<n;i++){prog('features',i,n);
    const bm=await createImageBitmap(images[i]),W=FW,H=Math.round(FW*bm.height/bm.width),oc=new OffscreenCanvas(W,H),cx=oc.getContext('2d');cx.drawImage(bm,0,0,W,H);bm.close&&bm.close();
    const src=c.matFromImageData(cx.getImageData(0,0,W,H)),gray=new c.Mat();c.cvtColor(src,gray,c.COLOR_RGBA2GRAY);src.delete();
    const kp=new c.KeyPointVector(),des=new c.Mat(),none=new c.Mat();orb.detectAndCompute(gray,none,kp,des);gray.delete();none.delete();
    const pts=new Float32Array(kp.size()*2);for(let k=0;k<kp.size();k++){const p=kp.get(k).pt;pts[2*k]=p.x;pts[2*k+1]=p.y;}kp.delete();
    feats.push({W,H,pts,des});}
  orb.delete();
  // which pairs to try: neighbours by the sensors' guess, or every pair without sensors
  const dir=q=>qrot(q,[0,0,-1]),pairs=[];
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){if(qs[i]&&qs[j]){const a=dir(qs[i]),b=dir(qs[j]);if(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]<Math.cos(85*Math.PI/180))continue;}pairs.push([i,j]);}
  const bf=new c.BFMatcher(c.NORM_HAMMING,false);let links=[];
  for(let p=0;p<pairs.length;p++){const [i,j]=pairs[p];prog('match',p,pairs.length);
    const A=feats[i],B=feats[j];if(A.des.rows<20||B.des.rows<20)continue;
    const mm=new c.DMatchVectorVector();bf.knnMatch(A.des,B.des,mm,2);const s=[],d=[];
    for(let k=0;k<mm.size();k++){const m=mm.get(k);if(m.size()<2)continue;const m0=m.get(0),m1=m.get(1);if(m0.distance<.78*m1.distance){s.push(A.pts[2*m0.queryIdx],A.pts[2*m0.queryIdx+1]);d.push(B.pts[2*m0.trainIdx],B.pts[2*m0.trainIdx+1]);}}
    mm.delete();if(s.length<2*20)continue;
    const sm=c.matFromArray(s.length/2,1,c.CV_32FC2,s),dm=c.matFromArray(d.length/2,1,c.CV_32FC2,d),mask=new c.Mat();
    const Hm=c.findHomography(sm,dm,c.RANSAC,TOL.ransac,mask);sm.delete();dm.delete();
    if(!Hm.empty()){const pa=[],pb=[];for(let k=0;k<mask.rows;k++)if(mask.data[k]){pa.push([s[2*k],s[2*k+1]]);pb.push([d[2*k],d[2*k+1]]);}
      if(pa.length>=15)links.push({i,j,pa,pb});}
    Hm.delete();mask.delete();}
  bf.delete();feats.forEach(f=>f.des.delete());
  // camera width: try a range of fields of view and keep the one that explains the matches best
  prog('solve',0,1);
  const W=FW,Hh=feats[0].H;
  const residual=fv=>{const FOVNOW=fv,f=fOf(feats[0],fv);let err=0,cnt=0;
    for(const L of links){const A=L.pa.map(p=>ray(p[0],p[1],feats[L.i].W,feats[L.i].H,fOf(feats[L.i],FOVNOW))),B=L.pb.map(p=>ray(p[0],p[1],feats[L.j].W,feats[L.j].H,fOf(feats[L.j],FOVNOW))),R=horn(A,B);
      for(let k=0;k<A.length;k++){const r=qrot(R,A[k]),b=B[k];err+=Math.acos(Math.min(1,r[0]*b[0]+r[1]*b[1]+r[2]*b[2])); cnt++;}}
    return cnt?err/cnt*f:1e9;}; // in pixels, so a narrower view cannot win just by making angles smaller
  let fov=hfov||64;
  // keep only pairs that a pure turn explains reasonably at the first guess, then find the camera's real angle of view from them
  { const f0=fOf(feats[0],fov);
    links=links.filter(L=>{if(L.pa.length<20)return false;const A=L.pa.map(p=>ray(p[0],p[1],feats[L.i].W,feats[L.i].H,fOf(feats[L.i],fov))),B=L.pb.map(p=>ray(p[0],p[1],feats[L.j].W,feats[L.j].H,fOf(feats[L.j],fov))),R=horn(A,B);
      const res=A.map((a,k)=>{const r=qrot(R,a),b=B[k];return Math.acos(Math.min(1,r[0]*b[0]+r[1]*b[1]+r[2]*b[2]))*f0;}).sort((x,y)=>x-y);return res[res.length>>1]<TOL.pre;}); }
  if(links.length>=3&&!fixFov){let best=1e9;for(let v=40;v<=100;v+=2){const e=residual(v);if(e<best){best=e;fov=v;}}
    const c0=fov;for(let v=c0-1.5;v<=c0+1.5;v+=.25){const e=residual(v);if(e<best){best=e;fov=v;}}}
  const FOVNOW=fov,f=fOf(feats[0],fov);
  // relative rotations: r_j = R_ij r_i  =>  Q_i = Q_j * R_ij
  let E=[];
  for(const L of links){const A=L.pa.map(p=>ray(p[0],p[1],feats[L.i].W,feats[L.i].H,fOf(feats[L.i],FOVNOW))),B=L.pb.map(p=>ray(p[0],p[1],feats[L.j].W,feats[L.j].H,fOf(feats[L.j],FOVNOW))),R=horn(A,B);
    // 1) the turn must explain the matched points to within ~2.5 pixels (a false match cannot)
    const res=A.map((a,k)=>{const r=qrot(R,a),b=B[k];return Math.acos(Math.min(1,r[0]*b[0]+r[1]*b[1]+r[2]*b[2]))*f;}).sort((x,y)=>x-y);
    if(res[res.length>>1]>TOL.strict)continue;
    // 2) it must not contradict the phone's sensors by a lot
    if(qs[L.i]&&qs[L.j]&&qang(R,qmul(qinv(qs[L.j]),qs[L.i]))>45*Math.PI/180)continue;
    if(A.length<20)continue;
    E.push({i:L.i,j:L.j,w:A.length,R});}
  // ---- with sensors: gravity gives tilt and roll reliably; only the heading drifts. Solve one heading correction per photo. ----
  if(qs.some(Boolean)&&!fullSolve){
    const yawOf=q=>{const v=qrot(q,[1,0,0]);return Math.atan2(-v[2],v[0]);}; // rotation about the vertical axis
    const tiltOf=q=>{const v=qrot(q,[0,1,0]);return Math.acos(Math.max(-1,Math.min(1,v[1])));}; // how far it leans off vertical
    const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
    // every pair that passed the first (loose) check counts here; gravity agreement does the strict filtering
    const EL=links.map(L=>({i:L.i,j:L.j,w:L.pa.length,R:horn(L.pa.map(p=>ray(p[0],p[1],feats[L.i].W,feats[L.i].H,fOf(feats[L.i],FOVNOW))),L.pb.map(p=>ray(p[0],p[1],feats[L.j].W,feats[L.j].H,fOf(feats[L.j],FOVNOW))))}));
    let M=[];for(const e of EL){const S_i=qs[e.i],S_j=qs[e.j];if(!S_i||!S_j)continue;
      const D=qmul(qmul(S_j,e.R),qinv(S_i)); // = turn about the vertical by (d_i - d_j) if all is well
      if(tiltOf(D)>TOL.tilt*Math.PI/180)continue; // the content says something gravity disagrees with: near-object shift, skip it
      M.push({i:e.i,j:e.j,w:e.w,m:wrap(yawOf(D))});}
    const d=new Float64Array(n),hasY=new Array(n).fill(false);M.forEach(x=>{hasY[x.i]=hasY[x.j]=true;});
    const k0=hasY.indexOf(true);
    const solve=()=>{for(let it=0;it<300;it++)for(let i=0;i<n;i++){if(i===k0||!hasY[i])continue;let sx=0,sy=0;
      for(const x of M){if(x.i===i){sx+=x.w*Math.cos(d[x.j]+x.m);sy+=x.w*Math.sin(d[x.j]+x.m);}else if(x.j===i){sx+=x.w*Math.cos(d[x.i]-x.m);sy+=x.w*Math.sin(d[x.i]-x.m);}}
      if(sx||sy)d[i]=Math.atan2(sy,sx);}};
    solve();
    for(let r=0;r<3;r++){const before=M.length;M=M.filter(x=>Math.abs(wrap(d[x.i]-d[x.j]-x.m))<TOL.agree*Math.PI/180);if(M.length===before)break;hasY.fill(false);M.forEach(x=>{hasY[x.i]=hasY[x.j]=true;});solve();}
    // photos with no usable match borrow the correction of the nearest photo in shooting order (drift changes slowly)
    for(let i=0;i<n;i++)if(!hasY[i]){let b=-1,bd=1e9;for(let j=0;j<n;j++)if(hasY[j]&&Math.abs(j-i)<bd){bd=Math.abs(j-i);b=j;}if(b>=0)d[i]=d[b];}
    const out=qs.map((S,i)=>S?qnorm(qmul([0,Math.sin(d[i]/2),0,Math.cos(d[i]/2)],S)):[0,0,0,1]);
    // self-check: how well do neighbouring photos agree with what their content says? keep whichever is better
    const disagree=Q=>{const v=EL.map(e=>qang(Q[e.i],qmul(Q[e.j],e.R))*180/Math.PI).sort((a,b)=>a-b);return v.length?v[v.length>>1]:1e9;};
    const before=disagree(qs.map(q=>q||[0,0,0,1])),after=disagree(out),use=after<before*.8;
    return{qs:(use?out:qs.map(q=>q||[0,0,0,1])).map(q=>q.map(v=>+v.toFixed(6))),hfov:+(use?fov:(hfov||64)).toFixed(2),matched:hasY,pairs:M.length,mode:use?'heading':'sensors',check:{before:+before.toFixed(2),after:+after.toFixed(2)},E:E.map(e=>({i:e.i,j:e.j,w:e.w,R:e.R}))};
  }
  // starting point: sensors if we have them, otherwise grow from the first photo
  const Q=new Array(n).fill(null),has=new Array(n).fill(false);E.forEach(e=>{has[e.i]=has[e.j]=true;});
  const anyS=qs.some(Boolean);
  if(anyS){for(let i=0;i<n;i++)Q[i]=qs[i]?qs[i].slice():[0,0,0,1];}
  else{Q[0]=[0,0,0,1];let grew=true;while(grew){grew=false;for(const e of E){if(Q[e.j]&&!Q[e.i]){Q[e.i]=qmul(Q[e.j],e.R);grew=true;}else if(Q[e.i]&&!Q[e.j]){Q[e.j]=qmul(Q[e.i],qinv(e.R));grew=true;}}}
    for(let i=0;i<n;i++)if(!Q[i])Q[i]=[0,0,0,1];}
  // all together: every photo moves to agree with its matched neighbours (repeat until it settles)
  const settle=()=>{for(let it=0;it<150;it++){
    for(let i=0;i<n;i++){const L=[];
      for(const e of E){if(e.i===i)L.push([qmul(Q[e.j],e.R),e.w]);else if(e.j===i)L.push([qmul(Q[e.i],qinv(e.R)),e.w]);}
      if(L.length){if(!anyS&&i===0)continue;Q[i]=qavg(L);}}}};
  settle();
  // 3) drop any pair that still disagrees with the whole picture, then settle again
  for(let round=0;round<3;round++){const before=E.length;E=E.filter(e=>qang(Q[e.i],qmul(Q[e.j],e.R))<TOL.agree*Math.PI/180);if(E.length===before)break;settle();}
  has.fill(false);E.forEach(e=>{has[e.i]=has[e.j]=true;});
  // keep gravity and "front" from the sensors on average (the photos only know where they are relative to each other)
  // anchor: the first matched photo keeps exactly where the phone said it was (it defines "front"; gravity is reliable)
  if(anyS){const k0=Q.findIndex((q,i)=>has[i]&&qs[i]);
    if(k0>=0){const g=qmul(qs[k0],qinv(Q[k0]));for(let i=0;i<n;i++)if(has[i])Q[i]=qnorm(qmul(g,Q[i]));}
    for(let i=0;i<n;i++){if(has[i]||!qs[i])continue;let best=-1,bd=1e9;for(let j=0;j<n;j++)if(has[j]&&qs[j]&&Math.abs(j-i)<bd){bd=Math.abs(j-i);best=j;}
      if(best>=0)Q[i]=qnorm(qmul(qmul(Q[best],qinv(qs[best])),qs[i]));}}
  return{qs:Q.map(q=>q.map(v=>+v.toFixed(6))),hfov:+fov.toFixed(2),matched:has,pairs:E.length,E:E.map(e=>({i:e.i,j:e.j,w:e.w,R:e.R}))};
}
self.onmessage=async e=>{try{postMessage({result:await stitch(e.data)});}catch(err){postMessage({error:String(err&&err.message||err)});}};
