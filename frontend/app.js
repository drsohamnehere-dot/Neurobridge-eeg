const $ = id => document.getElementById(id);
let token = sessionStorage.getItem('nb_token') || '';
let files = [], studies = [], currentStudyId = null, toastTimer = null;
let pageSelections = new Map(), imagePreviewCache = new Map();
const API = String(window.NB_API_BASE || '/api').replace(/\/$/, '');
function authHeaders(extra={}) { return { ...extra, Authorization: `Bearer ${token}` }; }
async function api(path, options={}) {
  const headers = authHeaders(options.headers || {});
  const res = await fetch(API + path, { ...options, headers });
  if (res.status === 401) { logout(); throw new Error('Session expired. Sign in again.'); }
  const data = await res.json().catch(()=>({}));
  if (!res.ok) throw new Error(data.detail || `Server error (${res.status})`);
  return data;
}
function toast(message) { const el=$('toast'); el.textContent=message; el.classList.add('show'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.classList.remove('show'),2600); }
function fmtBytes(n) { if(n<1024)return `${n} B`; if(n<1048576)return `${(n/1024).toFixed(1)} KB`; if(n<1073741824)return `${(n/1048576).toFixed(2)} MB`; return `${(n/1073741824).toFixed(2)} GB`; }
function fmtDate(s) { try{return new Date(s).toLocaleString();}catch{return s||'—';} }
function extensionOf(file){const ext=String(file?.extension||'').trim().toLowerCase();if(ext)return ext.startsWith('.')?ext:'.'+ext;const match=String(file?.original_name||'').match(/\.[^.]+$/);return match?match[0].toLowerCase():'';}
function logout(){token='';sessionStorage.removeItem('nb_token');$('authModal').classList.remove('hidden');$('loginStatus').textContent='Please sign in.';}
async function login(){const code=$('accessCode').value;if(!code){$('loginStatus').textContent='Enter the access code.';return;} $('loginBtn').disabled=true;$('loginStatus').textContent='Checking access code…';try{const r=await fetch(API+'/auth/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({access_code:code})});const d=await r.json();if(!r.ok)throw new Error(d.detail||'Login failed');token=d.token;sessionStorage.setItem('nb_token',token);$('authModal').classList.add('hidden');$('accessCode').value='';$('loginStatus').textContent='';await refreshAll();toast('Connected to NeuroBridge backend');}catch(e){$('loginStatus').textContent=e.message;}finally{$('loginBtn').disabled=false;}}
function showTab(name){document.querySelectorAll('.step').forEach(x=>x.classList.toggle('active',x.dataset.tab===name));document.querySelectorAll('.tab-panel').forEach(x=>x.classList.add('hidden'));$('tab-'+name).classList.remove('hidden');if(name==='edf'&&token){refreshEdfFileSelect().catch(e=>{$('edfFileListStatus').textContent='Could not refresh the EDF/BDF list: '+e.message;});}}
document.querySelectorAll('.step').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
async function checkServer(){try{const r=await fetch(API+'/health');const d=await r.json();$('apiStatus').classList.add('online');$('apiStatus').innerHTML='<i></i> Backend online · v'+d.version;}catch{$('apiStatus').classList.remove('online');$('apiStatus').innerHTML='<i></i> Backend unavailable';}}
async function refreshStats(){try{const s=await api('/stats');$('statFiles').textContent=s.files;$('statStudies').textContent=s.studies;$('statAnalyses').textContent=s.analyses;$('statStorage').textContent=fmtBytes(s.bytes_stored);}catch(e){console.warn(e);}}
async function refreshStudies(){studies=await api('/studies');const sel=$('existingStudy');const previous=currentStudyId||sessionStorage.getItem('nb_study_id')||'';sel.innerHTML='<option value="">Create a new study…</option>'+studies.map(s=>`<option value="${s.id}">${esc(s.title)} · ${s.subject_code?esc(s.subject_code):s.id.slice(0,8)}</option>`).join('');if(previous&&studies.some(s=>s.id===previous)){currentStudyId=previous;sel.value=previous;}else if(!currentStudyId){sel.value='';}}
async function refreshEdfFileSelect(){files=await api('/files');renderFiles();fillSelects();const count=files.filter(f=>['.edf','.bdf'].includes(extensionOf(f))).length;const status=$('edfFileListStatus');if(status)status.textContent=count?`${count} EDF/BDF recording${count===1?'':'s'} found on the server.`:'No EDF/BDF recordings found on the server yet. Upload one here or from Waveforms, then tap Refresh file list.';return count;}
async function refreshFiles(){files=await api('/files');renderFiles();fillSelects();await renderImageUploads();}
async function refreshAll(){await checkServer();await Promise.all([refreshStats(),refreshStudies(),refreshFiles(),refreshModels()]);if(currentStudyId){try{const s=await api('/studies/'+currentStudyId);$('studyTitle').value=s.title||'';$('subjectCode').value=s.subject_code||'';$('studyDate').value=s.study_date||'';$('studyAge').value=s.age??'';$('studySex').value=s.sex||'';$('studyMontage').value=s.montage||'';$('studyRate').value=s.sampling_rate??'';$('studyContext').value=s.context||'';$('studySource').value=s.source_text||'';$('studySaveStatus').textContent='Loaded saved study · '+currentStudyId.slice(0,8);await loadSavedPageSelections();}catch(e){console.warn(e);}}}
function renderFiles(){
 const body=$('filesTable');
 if(!files.length){body.innerHTML='<tr><td colspan="6" class="empty">No uploads yet.</td></tr>';return;}
 body.innerHTML=files.map(f=>{
  const isEdf=['.edf','.bdf'].includes(extensionOf(f)),label=isEdf?'EDF Analyse':'Analyse';
  return `<tr><td><div class="file-name">${esc(f.original_name)}</div><div class="file-sub">${f.sha256.slice(0,12)}…</div></td><td>${esc(extensionOf(f).toUpperCase())}</td><td>${fmtBytes(f.size_bytes)}</td><td>${fmtDate(f.uploaded_at)}</td><td><span class="status-tag ${f.status==='error'?'error':''}">${esc(f.status)}</span></td><td><div class="row-actions"><button class="tiny-btn" data-analyse="${f.id}">${label}</button><button class="tiny-btn" data-download="${f.id}">Save</button></div></td></tr>`;
 }).join('');
 body.querySelectorAll('[data-analyse]').forEach(b=>b.addEventListener('click',()=>{
  const file=files.find(f=>f.id===b.dataset.analyse);
  if(file&&['.edf','.bdf'].includes(file.extension)){showTab('edf');$('edfAnalysisFile').value=file.id;$('analyseEdfBtn').click();}
  else{showTab('review');$('analysisFile').value=b.dataset.analyse;$('analyseBtn').click();}
 }));
 body.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>downloadFile(b.dataset.download)));
}
function fillSelects(){
 const oldAnalysis=$('analysisFile').value,oldPdf=$('pdfFile').value,oldPrediction=$('predictionFile').value,oldEdf=$('edfAnalysisFile')?.value||'';
 const edfFiles=files.filter(f=>['.edf','.bdf'].includes(extensionOf(f)));
 const otherFiles=files.filter(f=>!['.edf','.bdf'].includes(extensionOf(f)));
 const edfListStatus=$('edfFileListStatus');if(edfListStatus)edfListStatus.textContent=edfFiles.length?`${edfFiles.length} EDF/BDF recording${edfFiles.length===1?'':'s'} found on the server.`:'No EDF/BDF recordings found on the server yet. Upload one here or from Waveforms, then tap Refresh file list.';
 $('analysisFile').innerHTML='<option value="">Select a file…</option>'+otherFiles.map(f=>`<option value="${f.id}">${esc(f.original_name)} (${fmtBytes(f.size_bytes)})</option>`).join('');
 $('edfAnalysisFile').innerHTML='<option value="">Select an EDF/BDF recording…</option>'+edfFiles.map(f=>`<option value="${f.id}">${esc(f.original_name)} · ${fmtBytes(f.size_bytes)}</option>`).join('');
 $('pdfFile').innerHTML='<option value="">Select PDF…</option>'+files.filter(f=>extensionOf(f)==='.pdf').map(f=>`<option value="${f.id}">${esc(f.original_name)}</option>`).join('');
 $('predictionFile').innerHTML='<option value="">Select CSV/TSV…</option>'+files.filter(f=>['.csv','.tsv','.txt'].includes(extensionOf(f))).map(f=>`<option value="${f.id}">${esc(f.original_name)}</option>`).join('');
 if(otherFiles.some(f=>f.id===oldAnalysis))$('analysisFile').value=oldAnalysis;
 if(files.some(f=>f.id===oldPdf&&extensionOf(f)==='.pdf'))$('pdfFile').value=oldPdf;
 if(files.some(f=>f.id===oldPrediction&&['.csv','.tsv','.txt'].includes(extensionOf(f))))$('predictionFile').value=oldPrediction;
 if(edfFiles.some(f=>f.id===oldEdf))$('edfAnalysisFile').value=oldEdf;
 else if(edfFiles.length)$('edfAnalysisFile').value=edfFiles[0].id;
}
async function renderImageUploads(){const area=$('imageSelectionGrid');const imgs=files.filter(f=>['.png','.jpg','.jpeg','.webp'].includes(extensionOf(f)));if(!imgs.length){area.innerHTML='<div class="empty">Uploaded PNG/JPG/WEBP images will appear here.</div>';return;}area.innerHTML='';for(const f of imgs){const card=document.createElement('div');card.className='image-select-card';const chosen=pageSelections.get(f.id)?.has(0)||false;card.innerHTML=`<img alt="${esc(f.original_name)}" src=""><div class="img-name">${esc(f.original_name)}</div><label><input type="checkbox" ${chosen?'checked':''}> Include in reviewer PDF</label>`;area.appendChild(card);const img=card.querySelector('img');if(imagePreviewCache.has(f.id)){img.src=imagePreviewCache.get(f.id);}else{const resp=await fetch(API+`/files/${f.id}/download`,{headers:authHeaders()});if(resp.ok){const url=URL.createObjectURL(await resp.blob());imagePreviewCache.set(f.id,url);img.src=url;}}const check=card.querySelector('input');check.addEventListener('change',e=>{if(!pageSelections.has(f.id))pageSelections.set(f.id,new Set());const set=pageSelections.get(f.id);if(e.target.checked){if(totalSelectedPages()>=8){e.target.checked=false;toast('Maximum 8 pages/images per study.');return;}set.add(0);}else set.delete(0);updateSelectedPageCount();});}}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function setUploadDiagnostic(message,state='info'){
  const el=$('uploadDiagnostic');
  if(!el)return;
  el.textContent=message;
  el.dataset.state=state;
}
function uploadFailureMessage(xhr,data,fileName){
  const detail=typeof data?.detail==='string'?data.detail:'';
  if(xhr.status===401){logout();return 'Your session expired. Sign in again, then retry the upload.';}
  if(xhr.status===413)return 'The server rejected this file because it is larger than its upload limit.';
  if(xhr.status===415)return detail||'This file type is not accepted by the server. Choose an EDF or BDF file.';
  if(xhr.status===422)return detail||'The server could not read the uploaded form. Please retry once.';
  if(xhr.status===502||xhr.status===503||xhr.status===504)return 'The hosting service is waking up or temporarily unavailable. Wait for the site to reconnect, then retry.';
  if(xhr.status===0)return 'The browser could not reach NeuroBridge. Check your internet connection and reload the page.';
  return detail||('Upload failed for '+fileName+' (HTTP '+xhr.status+').');
}
function uploadOne(file,index,total,batchState){
  return new Promise((resolve,reject)=>{
    if(!token){logout();reject(new Error('Please sign in before uploading.'));return;}
    const xhr=new XMLHttpRequest();
    const fd=new FormData();
    fd.append('file',file,file.name);
    if(currentStudyId)fd.append('study_id',currentStudyId);
    const id='q'+index;
    const queue=$('uploadQueue');
    const card=document.createElement('div');
    card.className='queue-item';
    card.innerHTML='<div><div class="queue-name">'+esc(file.name)+' · '+fmtBytes(file.size)+'</div><div class="mini-progress"><i id="'+id+'bar"></i></div></div><div class="queue-state" id="'+id+'state">Queued</div>';
    queue.appendChild(card);
    let lastLoaded=0,lastTime=Date.now(),speed=0;
    const stateEl=$(id+'state');
    const fail=(message)=>{
      stateEl.textContent='Failed: '+message;
      stateEl.style.color='var(--red)';
      card.title=message;
      setUploadDiagnostic(message,'error');
      reject(new Error(message));
    };
    xhr.timeout=600000;
    xhr.upload.addEventListener('loadstart',()=>{
      stateEl.textContent='Connecting…';
      setUploadDiagnostic('Connecting to the NeuroBridge upload endpoint…','info');
    });
    xhr.upload.addEventListener('progress',e=>{
      if(!e.lengthComputable)return;
      const pct=Math.min(100,Math.round(e.loaded/e.total*100));
      const fileLoaded=Math.min(file.size,Math.round((e.loaded/e.total)*file.size));
      $(id+'bar').style.width=pct+'%';
      stateEl.textContent=pct+'%';
      const now=Date.now();
      if(now-lastTime>=400){speed=(fileLoaded-lastLoaded)/((now-lastTime)/1000);lastLoaded=fileLoaded;lastTime=now;}
      batchState.loadedBefore[index]=fileLoaded;
      const loaded=batchState.loadedBefore.reduce((a,b)=>a+b,0);
      const overall=Math.min(100,Math.round(loaded/batchState.total*100));
      $('batchPercent').textContent=overall+'%';
      $('batchBar').style.width=overall+'%';
      $('batchBytes').textContent=fmtBytes(loaded)+' / '+fmtBytes(batchState.total);
      const eta=speed>0?(file.size-fileLoaded)/speed:null;
      $('batchEta').textContent=eta!==null?'Current file: '+fmtBytes(speed)+'/s · ~'+Math.ceil(eta)+'s remaining':'Transferring…';
      $('batchTitle').textContent='Uploading '+(index+1)+' of '+total;
      $('batchSub').textContent=file.name+' · '+pct+'% sent';
      setUploadDiagnostic('Uploading '+file.name+' · '+pct+'%','info');
    });
    xhr.upload.addEventListener('load',()=>{
      batchState.loadedBefore[index]=file.size;
      stateEl.textContent='Verifying…';
      const loaded=batchState.loadedBefore.reduce((a,b)=>a+b,0);
      $('batchBytes').textContent=fmtBytes(loaded)+' / '+fmtBytes(batchState.total);
      $('batchSub').textContent='Upload sent; server is verifying and storing the file…';
      setUploadDiagnostic('File sent. Waiting for server confirmation…','info');
    });
    xhr.addEventListener('load',()=>{
      let data={};
      try{data=JSON.parse(xhr.responseText||'{}');}catch{}
      if(xhr.status>=200&&xhr.status<300){
        $(id+'bar').style.width='100%';
        stateEl.textContent='Saved';
        stateEl.style.color='var(--green)';
        batchState.loadedBefore[index]=file.size;
        card.title='Stored successfully on the NeuroBridge server';
        setUploadDiagnostic(file.name+' was accepted and saved by the server.','success');
        resolve(data);
      }else{
        fail(uploadFailureMessage(xhr,data,file.name));
      }
    });
    xhr.addEventListener('error',()=>fail('Network error: the browser could not complete the upload request.'));
    xhr.addEventListener('timeout',()=>fail('Upload timed out after 10 minutes. Check the connection and retry.'));
    xhr.addEventListener('abort',()=>fail('Upload was cancelled.'));
    xhr.open('POST',API+'/files');
    xhr.setRequestHeader('Authorization','Bearer '+token);
    xhr.setRequestHeader('Accept','application/json');
    xhr.send(fd);
    setUploadDiagnostic('Upload request started for '+file.name+'.','info');
  });
}
async function uploadFiles(list){
  const selected=Array.from(list||[]);
  if(!selected.length){setUploadDiagnostic('No file was selected. Tap Choose files and select the EDF file.','warning');return;}
  if(!token){setUploadDiagnostic('Please sign in before uploading.','error');logout();return;}
  const supported=/\.(edf|bdf|csv|tsv|txt|json|pdf|png|jpe?g|webp)$/i;
  const invalid=selected.filter(f=>!supported.test(f.name));
  if(invalid.length){setUploadDiagnostic('Unsupported file extension: '+invalid.map(f=>f.name).join(', '),'error');}
  const tooLarge=selected.filter(f=>f.size>500*1024*1024);
  if(tooLarge.length){setUploadDiagnostic('File exceeds 500 MB: '+tooLarge.map(f=>f.name).join(', '),'error');}
  const empty=selected.filter(f=>!f.size);
  if(empty.length){setUploadDiagnostic('Empty file selected: '+empty.map(f=>f.name).join(', '),'error');}
  const valid=selected.filter(f=>supported.test(f.name)&&f.size>0&&f.size<=500*1024*1024);
  if(!valid.length){toast('No usable files selected. Check the upload message below.');return;}
  const total=valid.reduce((a,f)=>a+f.size,0);
  $('uploadQueue').innerHTML='';
  $('batchProgress').classList.remove('hidden');
  $('batchTitle').textContent='Preparing uploads';
  $('batchSub').textContent=valid.length+' file(s) · '+fmtBytes(total)+' total';
  $('batchPercent').textContent='0%';
  $('batchBar').style.width='0%';
  $('batchBytes').textContent='0 B / '+fmtBytes(total);
  const batch={total,loadedBefore:valid.map(()=>0)};
  let ok=0;
  const failures=[];
  setUploadDiagnostic('Preparing '+valid.length+' file(s) for upload…','info');
  for(let i=0;i<valid.length;i++){
    try{await uploadOne(valid[i],i,valid.length,batch);ok++;}
    catch(e){failures.push(valid[i].name+': '+e.message);toast(e.message);}
  }
  $('batchTitle').textContent='Upload batch finished · '+ok+'/'+valid.length+' saved';
  $('batchSub').textContent=failures.length?'Some files were not saved. See the error beside the failed file.':'All files were accepted by the server.';
  const loaded=batch.loadedBefore.reduce((a,b)=>a+b,0);
  $('batchPercent').textContent=Math.round(loaded/total*100)+'%';
  $('batchBar').style.width=Math.round(loaded/total*100)+'%';
  try{await refreshAll();}
  catch(e){setUploadDiagnostic('Upload finished, but the file list could not refresh: '+e.message,'error');}
  if(failures.length){
    setUploadDiagnostic(ok+' of '+valid.length+' file(s) saved. '+failures.join(' | '),'error');
    toast(ok+' of '+valid.length+' file(s) saved; upload errors need attention.');
  }else{
    setUploadDiagnostic(ok+' file(s) uploaded and confirmed by the NeuroBridge server.','success');
    toast(ok+' file(s) saved to backend');
  }
}
$('eegFileInput').addEventListener('change',async e=>{const selected=Array.from(e.target.files||[]);if(!selected.length){setUploadDiagnostic('No EEG file was selected. Tap Choose EDF/BDF and select an .edf or .bdf recording.','warning');return;}const bad=selected.filter(f=>!(/\.(edf|bdf)$/i.test(f.name)));if(bad.length){setUploadDiagnostic('The EEG picker accepts EDF/BDF recordings only: '+bad.map(f=>f.name).join(', '),'error');e.target.value='';return;}setUploadDiagnostic('Selected EEG recording: '+selected.map(f=>f.name).join(', '),'info');try{await uploadFiles(selected);}catch(err){setUploadDiagnostic('EEG upload did not start: '+(err?.message||String(err)),'error');toast(err?.message||'EEG upload failed');}finally{e.target.value='';}});
$('fileInput').addEventListener('change',async e=>{
  const selected=Array.from(e.target.files||[]);
  if(!selected.length){setUploadDiagnostic('No file was selected.','warning');return;}
  setUploadDiagnostic('Selected '+selected.length+' file(s): '+selected.map(f=>f.name).join(', '),'info');
  try{await uploadFiles(selected);}
  catch(err){setUploadDiagnostic('Upload did not start: '+(err?.message||String(err)),'error');toast(err?.message||'Upload failed');}
  finally{e.target.value='';}
});
const dz=$('dropZone');['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.add('dragging');}));['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{e.preventDefault();dz.classList.remove('dragging');}));dz.addEventListener('drop',e=>uploadFiles(e.dataTransfer.files));
document.addEventListener('paste',e=>{const items=Array.from(e.clipboardData?.items||[]);const imgs=items.filter(i=>i.type.startsWith('image/')).map((i,n)=>i.getAsFile()).filter(Boolean).map((f,i)=>new File([f],`pasted-waveform-${Date.now()}-${i}.png`,{type:f.type||'image/png'}));if(imgs.length)uploadFiles(imgs);});$('pasteHint').addEventListener('click',()=>toast('Copy a waveform screenshot, then paste it anywhere on this page.'));
async function downloadFile(id){try{const r=await fetch(API+`/files/${id}/download`,{headers:authHeaders()});if(!r.ok)throw new Error('Download failed');const blob=await r.blob();const f=files.find(x=>x.id===id);const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=f?.original_name||'eeg-file';a.click();URL.revokeObjectURL(a.href);}catch(e){toast(e.message);}}
$('existingStudy').addEventListener('change',async()=>{const id=$('existingStudy').value;currentStudyId=id||null;if(id){sessionStorage.setItem('nb_study_id',id);try{const s=await api('/studies/'+id);$('studyTitle').value=s.title||'';$('subjectCode').value=s.subject_code||'';$('studyDate').value=s.study_date||'';$('studyAge').value=s.age??'';$('studySex').value=s.sex||'';$('studyMontage').value=s.montage||'';$('studyRate').value=s.sampling_rate??'';$('studyContext').value=s.context||'';$('studySource').value=s.source_text||'';$('studySaveStatus').textContent='Loaded saved study · '+id.slice(0,8);await loadSavedPageSelections();}catch(e){toast(e.message);}}else{clearStudyForm();}});
function clearStudyForm(){currentStudyId=null;pageSelections.clear();updateSelectedPageCount();renderImageUploads();sessionStorage.removeItem('nb_study_id');$('studyTitle').value='';$('subjectCode').value='';$('studyDate').value='';$('studyAge').value='';$('studySex').value='';$('studyMontage').value='';$('studyRate').value='';$('studyContext').value='';$('studySource').value='';$('studySaveStatus').textContent='New study — not saved yet.';}
$('newStudy').addEventListener('click',()=>{$('existingStudy').value='';clearStudyForm();toast('New study draft opened');});
$('attachUnassigned').addEventListener('click',async()=>{if(!currentStudyId){toast('Save or open a study first.');return;}try{const r=await api(`/studies/${currentStudyId}/attach-unassigned`,{method:'POST'});toast(`${r.attached} unassigned file(s) attached to this study`);await refreshFiles();}catch(e){toast(e.message);}});
$('saveStudy').addEventListener('click',async()=>{try{const body={id:currentStudyId,title:$('studyTitle').value||'Untitled EEG study',subject_code:$('subjectCode').value||null,study_date:$('studyDate').value||null,age:$('studyAge').value?Number($('studyAge').value):null,sex:$('studySex').value||null,montage:$('studyMontage').value||null,sampling_rate:$('studyRate').value?Number($('studyRate').value):null,context:$('studyContext').value||null,source_text:$('studySource').value||$('quickNote').value||null};const s=await api('/studies',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});currentStudyId=s.id;sessionStorage.setItem('nb_study_id',s.id);$('studySaveStatus').textContent='Saved · '+s.id.slice(0,8);toast('Study details saved');await refreshAll();}catch(e){$('studySaveStatus').textContent=e.message;}});
async function runAnalysis(){const id=$('analysisFile').value;if(!id){toast('Choose a file first.');return;}$('analyseBtn').disabled=true;$('analysisStatus').textContent='Analysing file on server…';$('analysisOutput').classList.add('hidden');try{const r=await api(`/files/${id}/analyse?ocr=${$('enableOcr')?.checked===false?'false':'true'}`,{method:'POST'});renderAnalysis(r,'analysis');$('analysisStatus').textContent='Analysis saved to backend.';await refreshAll();}catch(e){$('analysisStatus').textContent=e.message;}finally{$('analyseBtn').disabled=false;}}
$('analyseBtn').addEventListener('click',runAnalysis);
function renderAnalysis(r,prefix='analysis'){const s=r.summary;$(prefix+'Output').classList.remove('hidden');$(prefix+'Type').textContent=s.kind||'Technical summary';const items=[];for(const [k,v] of Object.entries(s)){if(['channel_metrics','limitations','columns','channel_names','eeg_channels','annotations','label_classes'].includes(k))continue;if(typeof v==='string'||typeof v==='number'||typeof v==='boolean')items.push([k.replaceAll('_',' '),typeof v==='number'?Number(v.toFixed?.(3)??v):v]);} $(prefix+'Metrics').innerHTML=items.slice(0,12).map(([k,v])=>`<div class="metric"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');let lines=[];if(s.channel_metrics){lines.push('CHANNEL METRICS');for(const [ch,m] of Object.entries(s.channel_metrics)){lines.push(`${ch}: mean ${m.mean.toFixed(3)} · SD ${m.std.toFixed(3)} · RMS ${m.rms.toFixed(3)} · range ${m.min.toFixed(2)} to ${m.max.toFixed(2)}`);if(m.band_power)lines.push('  Exploratory band power: '+Object.entries(m.band_power).map(([k,v])=>`${k}=${v===null?'n/a':Number(v).toPrecision(3)}`).join(' | '));}}if(s.signal_review){const sr=s.signal_review;lines.push('QUANTITATIVE EEG SIGNAL SCREENING · RESEARCH ONLY');lines.push('Analysis window: '+sr.analysis_window_start_seconds+'–'+(sr.analysis_window_start_seconds+sr.analysis_window_seconds)+' s of '+sr.recording_duration_seconds+' s total');lines.push('Sampling rate: '+sr.sampling_rate_hz+' Hz · Channels processed: '+sr.channels_processed+' · Usable: '+sr.usable_channels);lines.push('Channels with quality flags: '+sr.channels_with_quality_flags+' · Generic transient candidates: '+sr.transient_candidate_count);for(const ch of (sr.channels||[])){if(!ch.usable){lines.push(ch.channel+': not usable · '+(ch.reason||''));continue;}lines.push(ch.channel+': RMS '+ch.rms_uV+' µV · peak-to-peak '+ch.peak_to_peak_uV+' µV · dominant frequency '+(ch.dominant_frequency_hz??'n/a')+' Hz');lines.push('  Relative band power: '+Object.entries(ch.relative_band_power_pct||{}).map(([k,v])=>k+' '+(v===null?'n/a':v+'%')).join(' | '));if(ch.quality_flags?.length)lines.push('  Quality flags: '+ch.quality_flags.join(', '));}if(sr.transient_candidates_top25?.length){lines.push('GENERIC TRANSIENT CANDIDATES FOR MANUAL REVIEW (NOT SPIKE/SEIZURE DETECTIONS)');for(const item of sr.transient_candidates_top25)lines.push(item.channel+' at '+item.time_seconds+' s · '+item.amplitude_uV+' µV');}lines.push(sr.interpretation||'');lines.push(sr.coverage_note||'');lines.push(sr.thresholds_note||'');}if(s.label_classes)lines.push('Supplied task labels: '+s.label_classes.join(', '));if(s.report_wording_review){const review=s.report_wording_review;lines.push('PDF REPORT WORDING REVIEW');if(review.items?.length){for(const item of review.items){lines.push(item.finding.replaceAll('_',' ')+': '+item.status.replaceAll('_',' '));for(const ev of (item.evidence||[]).slice(0,3))lines.push('  Page '+ev.page+': '+ev.wording);}}else{lines.push('No matching report phrases were identified in the extracted text.');}lines.push(review.note||'');}if(s.ocr_status)lines.push('OCR status: '+s.ocr_status+' · pages with OCR text: '+(s.ocr_pages_with_text??0));if(s.limitations)lines.push('LIMITATIONS\n• '+s.limitations.join('\n• '));if(r.extracted_text)lines.push('EXTRACTED PDF TEXT (first 12,000 characters)\n'+r.extracted_text);$(prefix+'Text').textContent=lines.join('\n\n')||'No additional summary text.';}
async function loadEdfWaveform(){
 const id=$('edfAnalysisFile').value;
 if(!id){$('edfWaveformStatus').textContent='Select an EDF/BDF recording first.';return;}
 const start=Math.max(0,Number($('edfStartSeconds').value)||0);
 $('loadEdfWaveform').disabled=true;
 $('edfWaveformStatus').textContent='Loading raw EEG samples…';
 try{
  const channelLimit=Number($('edfChannelLimit').value)||24;
  const preview=await api(`/files/${id}/edf-preview?start_seconds=${encodeURIComponent(start)}&duration_seconds=10&max_channels=${channelLimit}`);
  drawEdfWaveform(preview);
  const channelNote=preview.channels_displayed<channelLimit
   ? `This file contains only ${preview.available_scalp_channel_count} recognised scalp EEG channels; all available channels are shown.`
   : `Requested ${channelLimit}-channel capacity.`;
  $('edfWaveformStatus').textContent=`Showing ${preview.channels_displayed} of ${preview.available_scalp_channel_count} available scalp channels · ${preview.start_seconds}–${(preview.start_seconds+preview.duration_seconds).toFixed(1)} s · preview resampled to ${preview.preview_sampling_rate_hz} Hz. ${channelNote} ${preview.note}`;
  $('edfStartSeconds').max=Math.max(0,preview.recording_duration_seconds-10);
 }catch(e){$('edfWaveformStatus').textContent='Waveform preview failed: '+e.message;}
 finally{$('loadEdfWaveform').disabled=false;}
}
function drawEdfWaveform(preview){
 const canvas=$('edfWaveformCanvas'),rect=canvas.getBoundingClientRect(),width=Math.max(480,Math.floor(rect.width||900));
 const channels=preview.channels||[],height=Math.max(260,channels.length*58+36),dpr=Math.max(1,window.devicePixelRatio||1);
 canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.height=height+'px';
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#071321';ctx.fillRect(0,0,width,height);
 const left=64,right=14,top=16,bottom=28,plotW=width-left-right,plotH=height-top-bottom;
 ctx.font='11px system-ui';ctx.textBaseline='middle';ctx.strokeStyle='rgba(143,177,199,.22)';
 for(let sec=0;sec<=10;sec++){const x=left+plotW*sec/10;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+plotH);ctx.stroke();ctx.fillStyle='#91aabd';ctx.textAlign='center';ctx.fillText((preview.start_seconds+sec).toFixed(0)+'s',x,height-12);}
 const colors=['#58d5db','#f0b36d','#99a7ff','#89d3a4','#e99ab4','#b8a4ed','#e2d879','#82c8ef'];
 channels.forEach((channel,index)=>{
  const y0=top+plotH*(index+.5)/channels.length,band=plotH/channels.length;
  ctx.strokeStyle='rgba(143,177,199,.18)';ctx.beginPath();ctx.moveTo(left,y0);ctx.lineTo(width-right,y0);ctx.stroke();
  ctx.fillStyle='#d6e5ef';ctx.textAlign='right';ctx.fillText(channel.name,left-9,y0);
  const vals=channel.values_uv||[],abs=vals.map(v=>Math.abs(v)).sort((a,b)=>a-b),p95=abs.length?abs[Math.min(abs.length-1,Math.floor(abs.length*.95))]:1;
  const scale=(band*.38)/Math.max(20,p95);ctx.strokeStyle=colors[index%colors.length];ctx.lineWidth=1.15;ctx.beginPath();
  vals.forEach((v,j)=>{const x=left+plotW*(j/Math.max(1,vals.length-1)),y=y0-v*scale;if(j===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
 });
 ctx.fillStyle='#8faabd';ctx.textAlign='left';ctx.fillText('Amplitude automatically scaled per channel (µV)',left,8);
}
async function runEdfAnalysis(){
 const id=$('edfAnalysisFile').value;
 if(!id){$('edfAnalysisStatus').textContent='Upload an EDF/BDF recording first, then select it here.';return;}
 const button=$('analyseEdfBtn');button.disabled=true;
 const channelLimit=Number($('edfChannelLimit').value)||24;
 $('edfAnalysisStatus').textContent=`Reading EDF/BDF header and analysing up to ${channelLimit} scalp channels in the initial EEG window…`;
 $('edfAnalysisOutput').classList.add('hidden');
 try{
  const result=await api(`/files/${id}/analyse?ocr=false&channel_limit=${channelLimit}`,{method:'POST'});
  if(result.summary?.kind!=='edf_bdf')throw new Error('The selected file was not recognised as EDF/BDF.');
  renderAnalysis(result,'edfAnalysis');$('edfAnalysisOutput').classList.remove('hidden');$('edfStartSeconds').value='0';
  const processed=result.summary?.analysis_channels_processed??result.summary?.signal_review?.channels_processed??0;
  $('edfAnalysisStatus').textContent=`Analysis processed ${processed} of ${result.summary?.available_scalp_channel_count??processed} available scalp EEG channels. Loading waveform preview…`;
  await loadEdfWaveform();$('edfAnalysisStatus').textContent=`EDF/BDF analysis saved · ${processed} scalp channels processed (maximum selected: ${channelLimit}).`;await refreshAll();
 }catch(e){$('edfAnalysisStatus').textContent='EDF/BDF analysis failed: '+e.message;}
 finally{button.disabled=false;}
}
$('edfDirectFileInput').addEventListener('change',async e=>{
 const selected=Array.from(e.target.files||[]);
 if(!selected.length){$('edfUploadStatus').textContent='No file selected. Tap Choose EDF/BDF file and select a recording from Files.';return;}
 const invalid=selected.filter(f=>!(/\.(edf|bdf)$/i.test(f.name)));
 if(invalid.length){$('edfUploadStatus').textContent='Unsupported file. Please choose an .edf or .bdf recording.';toast('Please select an EDF or BDF file.');e.target.value='';return;}
 const file=selected[0];
 if(file.size>500*1024*1024){$('edfUploadStatus').textContent='This file is larger than the 500 MB upload limit.';toast('File exceeds 500 MB.');e.target.value='';return;}
 if(!file.size){$('edfUploadStatus').textContent='This file is empty. Please choose a valid EDF/BDF recording.';e.target.value='';return;}
 $('edfUploadStatus').textContent='Uploading '+file.name+' ('+fmtBytes(file.size)+')… Keep this page open until upload finishes.';
 try{
  await uploadFiles([file]);
  await refreshEdfFileSelect();
  const uploaded=files.find(f=>f.original_name===file.name&&['.edf','.bdf'].includes(extensionOf(f)));
  if(uploaded){
   $('edfAnalysisFile').value=uploaded.id;
   $('edfUploadStatus').textContent='Upload complete: '+file.name+'. It is saved on the NeuroBridge server and selected below. Tap Run EDF/BDF analysis.';
   toast('EDF/BDF recording uploaded successfully.');
  }else{
   $('edfUploadStatus').textContent='Upload was attempted, but the recording is not visible in the server file list. Check the upload message on the Upload Files tab and retry if needed.';
  }
 }catch(err){
  $('edfUploadStatus').textContent='Upload failed: '+(err?.message||String(err));
  toast(err?.message||'EDF upload failed');
 }finally{e.target.value='';}
});
$('edfChooseFileBtn').addEventListener('click',()=> $('edfDirectFileInput').click());
$('refreshEdfFilesBtn').addEventListener('click',async()=>{const b=$('refreshEdfFilesBtn');b.disabled=true;try{const count=await refreshEdfFileSelect();toast(count?`${count} EDF/BDF recording(s) found`:'No EDF/BDF recordings found');}catch(e){$('edfFileListStatus').textContent='Could not refresh the EDF/BDF list: '+e.message;toast(e.message);}finally{b.disabled=false;}});
$('analyseEdfBtn').addEventListener('click',runEdfAnalysis);
$('loadEdfWaveform').addEventListener('click',loadEdfWaveform);

$('loadPdf').addEventListener('click',async()=>{const id=$('pdfFile').value;if(!id){toast('Select a PDF first.');return;}const area=$('pdfPages');area.innerHTML='<div class="muted small">Reading PDF page list…</div>';try{const r=await api(`/files/${id}/pdf-info`);area.innerHTML='';if(!pageSelections.has(id))pageSelections.set(id,new Set());for(const p of r.pages){const card=document.createElement('div');card.className='pdf-page';const checked=pageSelections.get(id).has(p.page);card.innerHTML=`<img alt="PDF page ${p.page}" src=""><label><input type="checkbox" data-pdf-page="${p.page}" ${checked?'checked':''}> Select page ${p.page}</label><span>${p.width} × ${p.height} pt · ${p.has_text?'Embedded text':'Scanned/image page'}</span>`;area.appendChild(card);const img=card.querySelector('img');const resp=await fetch(API+`/files/${id}/pdf-page/${p.page}`,{headers:authHeaders()});if(resp.ok)img.src=URL.createObjectURL(await resp.blob());card.querySelector('input').addEventListener('change',e=>{const set=pageSelections.get(id);if(e.target.checked){if(totalSelectedPages()>=8){e.target.checked=false;toast('Maximum 8 pages per study.');return;}set.add(p.page);}else set.delete(p.page);updateSelectedPageCount();});}if(!r.pages.length)area.innerHTML='<div class="empty">No pages found.</div>';updateSelectedPageCount();}catch(e){area.innerHTML=`<div class="small" style="color:var(--red)">${esc(e.message)}</div>`;}});
function totalSelectedPages(){let n=0;for(const set of pageSelections.values())n+=set.size;return n;}
function updateSelectedPageCount(){$('selectedPageCount').textContent=`${totalSelectedPages()} / 8`;}
async function loadSavedPageSelections(){pageSelections.clear();const rows=await api(`/studies/${currentStudyId}/pages`);for(const row of rows){if(!pageSelections.has(row.file_id))pageSelections.set(row.file_id,new Set());pageSelections.get(row.file_id).add(row.page_num);}updateSelectedPageCount();await renderImageUploads();}
$('saveSelectedPages').addEventListener('click',async()=>{if(!currentStudyId){toast('Save or open a study first.');showTab('details');return;}const selections=Array.from(pageSelections.entries()).filter(([id,set])=>set.size).map(([file_id,set])=>({file_id,pages:Array.from(set).sort((a,b)=>a-b)}));try{const r=await api(`/studies/${currentStudyId}/pages`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({selections})});toast(`${r.saved_pages} selected PDF page(s) saved to this study`);}catch(e){toast(e.message);}});
$('trainBtn').addEventListener('click',async()=>{const b=$('trainBtn');b.disabled=true;$('modelStatus').textContent='Building labelled trial features and fitting baseline SVM…';try{const r=await api('/ml/train',{method:'POST'});$('modelStatus').textContent=`Model trained · balanced accuracy ${(r.metrics.balanced_accuracy*100).toFixed(1)}% · ${r.metrics.trials_total} trials. See validation warning below.`;await refreshModels();toast('Research baseline model saved');}catch(e){$('modelStatus').textContent=e.message;}finally{b.disabled=false;}});
async function refreshModels(){try{const data=await api('/ml/models');const area=$('modelsList');const oldModel=$('predictionModel').value;$('predictionModel').innerHTML='<option value="">Select a saved model…</option>'+data.models.map(m=>`<option value="${m.id}">${m.id.slice(0,8)} · ${m.classes.join('/')}${m.stale?' · STALE':''}</option>`).join('');if(data.models.some(m=>m.id===oldModel))$('predictionModel').value=oldModel;if(!data.models.length){area.innerHTML='<div class="empty">No saved models yet. Upload labelled CSV/TSV trials and train a baseline.</div>';return;}area.innerHTML=data.models.map(m=>`<div class="model-item"><div><strong>Research baseline · ${m.id.slice(0,8)}</strong><p>${fmtDate(m.created_at)} · classes: ${m.classes.map(esc).join(', ')}</p><p class="${m.stale?'stale':''}">${m.stale?'Dataset changed — model is stale; retrain to include new CSV/TSV data.':'Model matches current dataset revision.'}</p></div><div class="model-metric">Balanced accuracy ${(m.metrics.balanced_accuracy*100).toFixed(1)}%<div class="small muted">Test set: ${m.metrics.trials_test} trials</div></div></div>`).join('');}catch(e){console.warn(e);}}
$('predictBtn').addEventListener('click',async()=>{const mid=$('predictionModel').value,fid=$('predictionFile').value;if(!mid||!fid){$('predictionStatus').textContent='Select a saved model and an input file.';return;}const b=$('predictBtn');b.disabled=true;$('predictionStatus').textContent='Running trial-level predictions…';$('predictionOutput').textContent='';try{const r=await api(`/ml/predict/${mid}/${fid}`,{method:'POST'});$('predictionStatus').textContent=`Predicted ${r.predictions.length} trial(s).`;$('predictionOutput').textContent=r.predictions.map(p=>`${p.trial_id} → ${p.predicted_task_label} · ${Object.entries(p.probabilities).map(([k,v])=>`${k} ${(v*100).toFixed(1)}%`).join(' | ')}`).join('\n')+'\n\n'+r.warning;}catch(e){$('predictionStatus').textContent=e.message;}finally{b.disabled=false;}});
$('generateReport').addEventListener('click',async()=>{
 if(!currentStudyId){toast('Save or open the correct study first.');showTab('details');return;}
 const b=$('generateReport');b.disabled=true;
 try{
  await refreshFiles();
  const attachChoice=$('attachUnassignedForReport');
  const unassigned=files.filter(f=>!f.study_id);
  if(attachChoice?.checked&&unassigned.length){
   $('reportStatus').textContent='Attaching '+unassigned.length+' unassigned upload(s) to the selected study…';
   const linked=await api(`/studies/${currentStudyId}/attach-unassigned`,{method:'POST'});
   $('reportStatus').textContent=linked.attached+' file(s) attached. Preparing any EDF/BDF recordings and building the report…';
   await refreshFiles();
  }else{
   $('reportStatus').textContent='Preparing attached recordings and building the comprehensive report…';
  }
  const res=await fetch(API+`/studies/${currentStudyId}/report.pdf`,{headers:authHeaders()});
  if(!res.ok){const d=await res.json().catch(()=>({}));throw new Error(d.detail||'Could not generate report');}
  const blob=await res.blob();const url=URL.createObjectURL(blob);const link=document.createElement('a');
  link.href=url;link.download=`NeuroBridge_${currentStudyId.slice(0,8)}_EEG_review.pdf`;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1500);
  $('reportStatus').textContent='Comprehensive technical report generated. Check the file-status and clinical-review sections in the PDF.';
  toast('EEG review PDF generated');
 }catch(e){$('reportStatus').textContent='Report generation failed: '+e.message;toast(e.message);}
 finally{b.disabled=false;}
});
$('refreshBtn').addEventListener('click',()=>refreshAll().then(()=>toast('Server data refreshed')));$('reloadFiles').addEventListener('click',()=>refreshFiles());async function attachUnassignedToActiveStudy(button){if(!currentStudyId){toast('Save or open a study first.');showTab('details');return;}button.disabled=true;try{const r=await api(`/studies/${currentStudyId}/attach-unassigned`,{method:'POST'});await refreshAll();toast(r.attached?`${r.attached} unassigned file(s) attached to this study`:'No unassigned files found.');}catch(e){toast(e.message);}finally{button.disabled=false;}}
$('attachUnassigned').addEventListener('click',()=>attachUnassignedToActiveStudy($('attachUnassigned')));
if($('attachUnassignedFromDetails'))$('attachUnassignedFromDetails').addEventListener('click',()=>attachUnassignedToActiveStudy($('attachUnassignedFromDetails')));$('logoutBtn').addEventListener('click',logout);$('loginBtn').addEventListener('click',login);$('accessCode').addEventListener('keydown',e=>{if(e.key==='Enter')login();});
if(token){$('authModal').classList.add('hidden');refreshAll().catch(()=>logout());}else{$('authModal').classList.remove('hidden');checkServer();}
setInterval(()=>{if(token){refreshStats();refreshFiles();refreshModels();}},15000);
