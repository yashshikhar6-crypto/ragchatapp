const chatBox = document.getElementById('chatBox');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');

// 🧠 Add message bubble
function addMessage(text, sender = 'bot') {
  const msg = document.createElement('div');
  msg.className = `message ${sender}-msg`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

// ✨ Typing effect for bot messages
async function typeMessage(text) {
  const msg = document.createElement('div');
  msg.className = 'message bot-msg';
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatBox.appendChild(msg);

  let i = 0;
  const interval = setInterval(() => {
    bubble.textContent = text.slice(0, i++);
    chatBox.scrollTop = chatBox.scrollHeight;
    if (i > text.length) clearInterval(interval);
  }, 20);
}

// 📂 File upload handler
uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return alert('Please select a .pdf or .txt file first.');

  uploadBtn.disabled = true;
  uploadStatus.textContent = '📤 Uploading & indexing...';
  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (res.ok) {
      uploadStatus.textContent = data.message || 'Upload complete.';
      addMessage('✅ ' + (data.message || 'File uploaded successfully.'), 'bot');
    } else {
      uploadStatus.textContent = data.error || 'Upload failed';
      addMessage('⚠️ ' + (data.error || 'Upload failed'), 'bot');
    }
  } catch (e) {
    uploadStatus.textContent = '⚠️ Network error';
    addMessage('⚠️ Network error: ' + e.message, 'bot');
  } finally {
    uploadBtn.disabled = false;
  }
});

// 💬 Send question to backend
sendBtn.addEventListener('click', sendQuestion);
userInput.addEventListener('keydown', e => e.key === 'Enter' && sendQuestion());

async function sendQuestion() {
  const q = userInput.value.trim();
  if (!q) return;
  addMessage(q, 'user');
  userInput.value = '';

  const loading = document.createElement('div');
  loading.className = 'message bot-msg';
  loading.innerHTML = `<div class="avatar"></div><div class="msg-bubble">💭 Thinking...</div>`;
  chatBox.appendChild(loading);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    chatBox.removeChild(loading);
    if (res.ok) typeMessage(data.answer || 'No answer found.');
    else typeMessage('⚠️ ' + (data.error || data.answer || 'Error.'));
  } catch (e) {
    chatBox.removeChild(loading);
    typeMessage('⚠️ Network error: ' + e.message);
  }
}
