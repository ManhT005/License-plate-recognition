  const fileInput   = document.getElementById('file-input');
  const uploadZone  = document.getElementById('upload-zone');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImg  = document.getElementById('preview-img');
  const previewVideo = document.getElementById('preview-video');
  const removeBtn   = document.getElementById('remove-btn');
  const changeImgBtn= document.getElementById('change-img-btn');
  const predictBtn  = document.getElementById('predict-btn');
  const statusEl    = document.getElementById('status');
  const resultWrap  = document.getElementById('result-img-wrap');
  const resultPlaceholder = document.getElementById('result-placeholder');
  const platesList  = document.getElementById('plates-list');
  const debugToggle = document.getElementById('debug-toggle');
  const debugSection= document.getElementById('debug-section');
  const debugContent= document.getElementById('debug-content');

  let currentFile = null;

  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

  function handleFile(file) {
    if (!file) return;
    currentFile = file;
    const url = URL.createObjectURL(file);
    
    if (file.type.startsWith('video/')) {
        previewImg.style.display = 'none';
        previewVideo.src = url;
        previewVideo.style.display = 'block';
    } else {
        previewVideo.style.display = 'none';
        previewVideo.src = '';
        previewImg.src = url;
        previewImg.style.display = 'block';
    }

    previewWrap.style.display = 'block';
    changeImgBtn.style.display = 'block';
    predictBtn.disabled = false;
    clearStatus();
    platesList.innerHTML = '';
    resultWrap.innerHTML = '';
    resultWrap.appendChild(resultPlaceholder);
    resultPlaceholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3; margin-bottom:10px;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
      Sẵn sàng nhận diện
    `;
    uploadZone.style.display = 'none';
  }

  removeBtn.addEventListener('click', () => {
    currentFile = null;
    fileInput.value = '';
    previewWrap.style.display = 'none';
    changeImgBtn.style.display = 'none';
    predictBtn.disabled = true;
    platesList.innerHTML = '';
    resultWrap.innerHTML = '';
    resultWrap.appendChild(resultPlaceholder);
    
    previewVideo.style.display = 'none';
    previewVideo.src = '';
    previewImg.style.display = 'block';
    previewImg.src = '';

    resultPlaceholder.innerHTML = `
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3; margin-bottom:10px;"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
      Ảnh hoặc dòng thời gian video sẽ hiển thị ở đây
    `;
    debugContent.innerHTML = '<p style="text-align:center; color:var(--text-muted); font-style:italic;">Vui lòng thực hiện nhận diện ảnh trước để xem thông tin debug.</p>';
    clearStatus();
    uploadZone.style.display = 'block';
  });

  changeImgBtn.addEventListener('click', () => {
    fileInput.click();
  });

  uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) handleFile(file);
  });

  predictBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    const isVideo = currentFile.type.startsWith('video/');
    setStatus('info', '<span class="spinner"></span> Đang phân tích ' + (isVideo ? 'video' : 'hình ảnh') + '...');
    predictBtn.disabled = true;
    platesList.innerHTML = '';
    resultWrap.innerHTML = '';
    
    if(isVideo) {
         resultWrap.innerHTML = '<div style="padding: 30px 20px; text-align: center; color: var(--text-muted); line-height: 1.6;"><span class="spinner" style="width:24px; height:24px; margin-bottom:15px; border-color: rgba(255,255,255,0.1); border-top-color: var(--accent-1); display:block; margin: 0 auto 15px auto;"></span> Hệ thống đang quét toàn bộ Video <br/>Quá trình này có thể mất vài giây.</div>';
    } else {
         resultWrap.appendChild(resultPlaceholder);
         resultPlaceholder.innerHTML = '<span class="spinner" style="width:24px; height:24px; margin:0 0 10px 0; border-color: rgba(255,255,255,0.1); border-top-color: var(--accent-1);"></span> Đang xử lý bằng AI...';
    }

    const formData = new FormData();
    formData.append(isVideo ? 'video' : 'image', currentFile);
    const endpoint = isVideo ? '/predict_video' : '/predict';

    try {
      const res  = await fetch(endpoint, { method: 'POST', body: formData });
      const data = await res.json();

      if (data.error) {
        setStatus('error', '⚠ ' + data.error);
        if(!isVideo) resultPlaceholder.innerHTML = 'Có lỗi xảy ra trong quá trình xử lý.';
        else resultWrap.innerHTML = `<div style="padding: 20px; color: var(--error); text-align: center;">⚠ ${data.error}</div>`;
      } else if (isVideo) {
        // --- XỬ LÝ KẾT QUẢ VIDEO (DEDUPLICATED TIMELINE) ---
        resultWrap.innerHTML = `
            <div style="width: 100%; padding: 20px; background: rgba(16, 185, 129, 0.1); border-bottom: 1px solid var(--panel-border); text-align: center;">
                <h3 style="color: var(--success); font-size: 1.2rem; margin-bottom: 5px;">✔ Hoàn thành phân tích Video</h3>
                <p style="color: var(--text-muted); font-size: 0.9rem;">Tổng số xe (biển số độc lập) phát hiện: <strong style="color:#fff;">${data.total_unique}</strong></p>
            </div>
            <div id="video-timeline" style="width: 100%; max-height: 500px; overflow-y: auto; padding: 20px;"></div>
        `;
        
        const timelineWrap = document.getElementById('video-timeline');
        
        data.results.forEach((item) => {
            const timeBlock = document.createElement('div');
            timeBlock.style.marginBottom = '25px';
            timeBlock.style.background = 'rgba(0,0,0,0.3)';
            timeBlock.style.borderRadius = '12px';
            timeBlock.style.overflow = 'hidden';
            timeBlock.style.border = '1px solid var(--panel-border)';
            
            let innerHTML = `
                <div style="padding: 10px 15px; background: rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--accent-1); font-family: var(--font-mono); font-weight: bold;">🕒 Thời gian ${item.timestamp}</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted)">Phát hiện ${item.plates.length} xe mới</span>
                </div>
                <img src="${item.img_url}" style="width: 100%; max-height: 250px; object-fit: contain; background: #000; border-bottom: 1px solid rgba(255,255,255,0.05);" />
                <div style="padding: 15px; display: flex; flex-direction: column; gap: 10px;">
            `;
            
            item.plates.forEach((p) => {
              const cls = p.class.toLowerCase();
              innerHTML += `
                  <div class="plate-card" style="margin: 0; animation: none; opacity: 1; transform: none; background: rgba(255,255,255,0.03);">
                    <span class="plate-number">${p.text || '—'}</span>
                    <span class="plate-class ${cls}">${p.class}</span>
                  </div>
              `;
            });
            
            innerHTML += `</div>`;
            timeBlock.innerHTML = innerHTML;
            timelineWrap.appendChild(timeBlock);
        });

        setStatus('success', `✔ Phân tích xong! Đã lọc trùng lặp và ghi nhận ${data.total_unique} biển số khác nhau.`);
        debugContent.innerHTML = '<p style="text-align:center; color:var(--text-muted); font-style:italic;">Thông tin debug ký tự chi tiết chỉ hỗ trợ cho chế độ nhận diện ảnh tĩnh.</p>';
        
      } else {
        // --- XỬ LÝ KẾT QUẢ ẢNH (GIỮ NGUYÊN) ---
        resultWrap.innerHTML = `<img src="${data.img_url}" alt="Kết quả nhận diện" />`;
        data.plates.forEach((p, i) => {
          const cls = p.class.toLowerCase();
          const card = document.createElement('div');
          card.className = 'plate-card';
          card.style.animationDelay = `${i * 100}ms`;
          card.innerHTML = `
            <span class="plate-index">#${p.index}</span>
            <span class="plate-number">${p.text || '—'}</span>
            <span class="plate-class ${cls}">${p.class}</span>
          `;
          platesList.appendChild(card);
        });

        setStatus('success', `✔ Đã phát hiện thành công ${data.plates.length} biển số`);
        runDebug(currentFile);
      }
    } catch (err) {
      setStatus('error', '⚠ Lỗi kết nối tới máy chủ: ' + err.message);
    }

    predictBtn.disabled = false;
  });

  debugToggle.addEventListener('click', () => {
    debugToggle.classList.toggle('open');
    debugSection.classList.toggle('open');
  });

  async function runDebug(file) {
    const formData = new FormData();
    formData.append('image', file);
    try {
      const res  = await fetch('/debug', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) return;
      renderDebug(data);
    } catch {}
  }

  function renderDebug(data) {
    debugContent.innerHTML = '';
    data.forEach((plate, i) => {
      const card = document.createElement('div');
      card.className = 'debug-card';

      card.innerHTML = `
        <div style="margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 12px;">
          <h3 style="color:#fff; font-size:1.1rem; font-weight:600;">Biển số #${i+1} <span style="color:var(--text-muted); font-size:0.9rem; font-weight:400; margin-left:8px;">(${plate.plate_class} — ${plate.num_chars} ký tự)</span></h3>
        </div>
        
        <div class="debug-plate-imgs">
          <div>
            <p class="debug-label">Ảnh cắt nguyên bản (Crop)</p>
            <img src="${plate.plate_url}" alt="plate crop" />
          </div>
          <div>
            <p class="debug-label">Ảnh sau xử lý (Binary/Contrast)</p>
            <img src="${plate.binary_url}" alt="binary processed" />
          </div>
        </div>
        
        <p class="debug-label">Kết quả tách từng ký tự</p>
      `;

      const grid = document.createElement('div');
      grid.className = 'debug-grid';

      if (!plate.chars || plate.chars.length === 0) {
        grid.innerHTML = '<p style="color:var(--error); font-size:0.9rem; margin-top:10px;">Không thể tách được ký tự nào từ biển số này.</p>';
      } else {
        plate.chars.forEach(c => {
          const charItem = document.createElement('div');
          charItem.className = 'debug-char';
          charItem.innerHTML = `
            <img src="${c.path}" alt="${c.pred}" />
            <div class="char-pred">${c.pred}</div>
            <div class="char-conf">${c.conf}</div>
            <div class="char-size">${c.size}</div>
          `;
          grid.appendChild(charItem);
        });
      }

      card.appendChild(grid);
      debugContent.appendChild(card);
    });
  }

  function setStatus(type, msg) {
    statusEl.className = type;
    statusEl.innerHTML = msg;
    statusEl.style.display = 'block';
  }

  function clearStatus() {
    statusEl.style.display = 'none';
    statusEl.innerHTML = '';
  }

 /* ─── REALTIME FUNCTIONALITY ─────────────────────────── */

  const videoStream = document.getElementById('video-stream');
  const videoPlaceholder = document.getElementById('video-placeholder');
  const btnStart = document.getElementById('btn-start-realtime');
  const btnStop = document.getElementById('btn-stop-realtime');
  const statFps = document.getElementById('stat-fps');
  const statTime = document.getElementById('stat-time');
  const statFrames = document.getElementById('stat-frames');
  const resultsContainer = document.getElementById('realtime-results');
  const btnClearHistory = document.getElementById('btn-clear-history');
  
  let isRealtimeRunning = false;
  let pollingActive = false;


  if(btnClearHistory) {
      btnClearHistory.addEventListener('click', async () => {
          try {
              // Gọi lệnh xóa triệt để từ Server
              await fetch('/realtime/clear_history', { method: 'POST' });
          } catch (err) {
              console.warn("Lỗi khi xóa lịch sử:", err);
          }
          // Xóa giao diện tạm thời trong lúc chờ frame tiếp theo load về
          resultsContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem; text-align:center; padding:10px;">Chưa phát hiện</p>';
      });
  }

  btnStart.addEventListener('click', async () => {
    try {
      setStatus('info', '<span class="spinner"></span> Khởi động camera...');
      const res = await fetch('/realtime/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 0, fps_limit: 15 })
      });
      
      if (!res.ok) {
        const err = await res.json();
        setStatus('error', '⚠ ' + (err.error || 'Không thể khởi động'));
        return;
      }

      isRealtimeRunning = true;
      pollingActive = true;
      btnStart.disabled = true;
      btnStop.disabled = false;
      setStatus('success', '✔ Camera đang chạy');

      pollVideoFrame();
      pollStats();
    } catch (err) {
      setStatus('error', '⚠ Lỗi kết nối: ' + err.message);
    }
  });

  btnStop.addEventListener('click', async () => {
    btnStop.disabled = true; 
    setStatus('info', '<span class="spinner"></span> Đang dừng camera...');
    
    try {
      const res = await fetch('/realtime/stop', { method: 'POST' });
      if (res.ok) {
        stopRealtime();
        setStatus('success', '✔ Camera đã dừng');
      } else {
         setStatus('error', '⚠ Có lỗi khi dừng camera từ server');
         stopRealtime();
      }
    } catch (err) {
      setStatus('error', '⚠ Lỗi kết nối khi dừng: ' + err.message);
      stopRealtime(); 
    }
  });

  function stopRealtime() {
    isRealtimeRunning = false;
    pollingActive = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    
    setTimeout(() => {
        videoStream.src = '';
        videoStream.style.display = 'none';
        videoPlaceholder.style.display = 'flex';
    }, 500);
  }

  async function pollVideoFrame() {
    if (!isRealtimeRunning || !pollingActive) return;
    
    try {
      const res = await fetch('/video_frame');
      if (!res.ok) {
         if(res.status !== 500) throw new Error('Server ngắt kết nối stream');
      } else {
          const blob = await res.blob();
          if (blob.size > 0) {
              const url = URL.createObjectURL(blob);
              videoStream.style.display = 'block';
              videoPlaceholder.style.display = 'none';
              videoStream.onload = () => URL.revokeObjectURL(url);
              videoStream.src = url;
          }
      }
    } catch (err) {
      console.warn('Lấy frame thất bại:', err);
    }
    
    if (pollingActive) setTimeout(pollVideoFrame, 30);
  }

 async function pollStats() {
    if (!isRealtimeRunning || !pollingActive) return;

    try {
      const res = await fetch('/realtime/results');
      if (res.ok) {
          const data = await res.json();
          
          // Cập nhật các chỉ số FPS, Thời gian, Frame
          statFps.textContent = (data.fps || 0).toFixed(1);
          statTime.textContent = (data.processing_time || '0ms');
          statFrames.textContent = (data.frame_count || 0);

          // Lấy danh sách 'finalized_plates' từ Backend và hiển thị
          if (data.finalized_plates && data.finalized_plates.length > 0) {
              resultsContainer.innerHTML = [...data.finalized_plates].reverse()
                .map(p => `<div class="realtime-plate-item" style="color: #10b981;">✔ ${p}</div>`)
                .join('');
          } else if (data.finalized_plates && data.finalized_plates.length === 0) {
              // Nếu chưa có xe nào chốt sổ
              resultsContainer.innerHTML = '<p style="color:var(--text-muted); font-size:0.85rem; text-align:center; padding:10px;">Chưa có xe đi qua</p>';
          }
      }
    } catch (err) {
      console.warn('Lấy stats thất bại:', err);
    }

    if (pollingActive) setTimeout(pollStats, 500);
  }

  window.addEventListener('beforeunload', async () => {
    if (isRealtimeRunning) {
      try { await fetch('/realtime/stop', { method: 'POST', keepalive: true }); } catch {}
    }
  });