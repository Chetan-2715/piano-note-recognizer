const recordBtn = document.getElementById('record');
const notesEl = document.getElementById('notes');
let ws;
let audioCtx;
let processor;
let source;
let running = false;
let bufferQueue = [];
const CHUNK_SECONDS = 0.3; // 300ms
const WS_URL = 'ws://localhost:8000/ws';

function connectWS(sampleRate){
	ws = new WebSocket(WS_URL);
	ws.binaryType = 'arraybuffer';
	ws.onopen = () => {
		ws.send(JSON.stringify({type:'init', sampleRate: sampleRate}));
	};
	ws.onmessage = (ev) => {
		try{
			const obj = JSON.parse(ev.data);
			if(obj.type === 'pred'){
				updateNotes(obj.note);
			}
		}catch(e){}
	};
	ws.onclose = ()=>{ console.log('ws closed'); };
}

function updateNotes(note){
	// append with comma separation and keep short history (10)
	const cur = notesEl.textContent.trim();
	if(cur === '—' || cur === '' || cur === 'Listening...') notesEl.textContent = note;
	else {
		let arr = cur.split(',').map(s=>s.trim());
		if(arr[arr.length-1] !== note) arr.push(note);
		if(arr.length>10) arr = arr.slice(-10);
		notesEl.textContent = arr.join(', ');
	}
}

async function startRecording(){
	if(running) return stopRecording();
	notesEl.textContent = 'Listening...';
	running = true;
	recordBtn.classList.add('listening');
	audioCtx = new (window.AudioContext || window.webkitAudioContext)();
	const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
	source = audioCtx.createMediaStreamSource(stream);
	// Buffer size small to keep low latency
	const bufferSize = 4096;
	processor = audioCtx.createScriptProcessor(bufferSize, 1, 1);
	let sampleRate = audioCtx.sampleRate;
	connectWS(sampleRate);
	let samplesPerChunk = Math.floor(CHUNK_SECONDS * sampleRate);
	let ring = new Float32Array(0);
	processor.onaudioprocess = (evt) => {
		const input = evt.inputBuffer.getChannelData(0);
		// append to ring
		let newBuf = new Float32Array(ring.length + input.length);
		newBuf.set(ring,0);
		newBuf.set(input, ring.length);
		ring = newBuf;
		while(ring.length >= samplesPerChunk){
			const sendBuf = ring.slice(0, samplesPerChunk);
			ring = ring.slice(samplesPerChunk);
			// send as Float32Array raw bytes
			if(ws && ws.readyState === WebSocket.OPEN){
				ws.send(sendBuf.buffer);
			}
		}
	};
	source.connect(processor);
	processor.connect(audioCtx.destination);
}

function stopRecording(){
	running = false;
	recordBtn.classList.remove('listening');
	
	if(processor){ processor.disconnect(); processor.onaudioprocess = null; processor = null; }
	if(source){ source.disconnect(); source = null; }
	if(ws){ ws.close(); ws = null; }
	if(audioCtx){ audioCtx.close(); audioCtx = null; }
}

recordBtn.addEventListener('click', ()=>{
	if(!running) startRecording(); else stopRecording();
});
