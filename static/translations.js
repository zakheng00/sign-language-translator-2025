```javascript
// 共用的 TRANSLATIONS 對象，包含所有頁面的翻譯
const TRANSLATIONS = {
  index: {
    en: {
      title: "Sign Language Translator",
      subtitle: "Real-time translation for sign language and speech",
      welcome: "Welcome to the Sign Language Translation System",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
    },
    ms: {
      title: "Penterjemah Bahasa Isyarat",
      subtitle: "Terjemahan masa nyata untuk bahasa isyarat dan pertuturan",
      welcome: "Selamat Datang ke Sistem Penterjemahan Bahasa Isyarat",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan"
    }
  },
  liveTranslation: {
    en: {
      title: "Live Translation",
      subtitle: "Capture and translate sign language in real-time",
      startRecording: "Start Recording",
      stopRecording: "Stop Recording",
      transcribe: "Transcribe",
      statusDefault: "Click 'Start Recording' to begin",
      statusRecording: "Recording in progress...",
      statusSuccess: "Translation: {result}",
      statusError: "Error: {error}"
    },
    ms: {
      title: "Terjemahan Langsung",
      subtitle: "Tangkap dan terjemah bahasa isyarat dalam masa nyata",
      startRecording: "Mula Rakaman",
      stopRecording: "Henti Rakaman",
      transcribe: "Terjemah",
      statusDefault: "Klik 'Mula Rakaman' untuk memulakan",
      statusRecording: "Sedang merakam...",
      statusSuccess: "Terjemahan: {result}",
      statusError: "Ralat: {error}"
    }
  },
  roomMode: {
    en: {
      title: "Room Mode",
      subtitle: "Collaborate and translate in real-time",
      joinRoom: "Join Room",
      sendMessage: "Send Message",
      modeSign: "Sign Language Mode",
      modeSpeech: "Speech Mode",
      statusConnected: "Connected to room",
      statusDisconnected: "Disconnected. Attempting to reconnect...",
      statusMessageSent: "Message sent",
      statusError: "Error: {error}"
    },
    ms: {
      title: "Mod Bilik",
      subtitle: "Berkolaborasi dan terjemah dalam masa nyata",
      joinRoom: "Sertai Bilik",
      sendMessage: "Hantar Mesej",
      modeSign: "Mod Bahasa Isyarat",
      modeSpeech: "Mod Pertuturan",
      statusConnected: "Bersambung ke bilik",
      statusDisconnected: "Terputus. Sedang cuba menyambung semula...",
      statusMessageSent: "Mesej dihantar",
      statusError: "Ralat: {error}"
    }
  },
  speechToText: {
    en: {
      title: "Speech to Text",
      subtitle: "Record your audio and get instant text transcription",
      startRecording: "Start Recording",
      stopRecording: "Stop Recording",
      transcribe: "Transcribe",
      statusDefault: "Click 'Start Recording' to begin",
      statusRecording: "Recording in progress...",
      statusSuccess: "Transcription Result: {result}",
      statusError: "Error: {error}",
      statusConnected: "Connected to transcription service",
      statusDisconnected: "Disconnected. Attempting to reconnect..."
    },
    ms: {
      title: "Pertuturan ke Teks",
      subtitle: "Rakam audio anda dan dapatkan transkripsi teks segera",
      startRecording: "Mula Rakaman",
      stopRecording: "Henti Rakaman",
      transcribe: "Terjemah",
      statusDefault: "Klik 'Mula Rakaman' untuk memulakan",
      statusRecording: "Sedang merakam...",
      statusSuccess: "Keputusan Transkripsi: {result}",
      statusError: "Ralat: {error}",
      statusConnected: "Bersambung ke perkhidmatan transkripsi",
      statusDisconnected: "Terputus. Sedang cuba menyambung semula..."
    }
  },
  history: {
    en: {
      title: "Translation History",
      subtitle: "View and manage your translation records",
      noRecords: "No translation records found",
      deleteRecord: "Delete",
      statusLoading: "Loading history...",
      statusSuccess: "History loaded successfully",
      statusError: "Failed to load history: {error}",
      deleteSuccess: "Record deleted successfully",
      deleteError: "Failed to delete record: {error}"
    },
    ms: {
      title: "Sejarah Terjemahan",
      subtitle: "Lihat dan urus rekod terjemahan anda",
      noRecords: "Tiada rekod terjemahan ditemui",
      deleteRecord: "Padam",
      statusLoading: "Memuatkan sejarah...",
      statusSuccess: "Sejarah dimuatkan dengan jayanya",
      statusError: "Gagal memuatkan sejarah: {error}",
      deleteSuccess: "Rekod dipadamkan dengan jayanya",
      deleteError: "Gagal memadam rekod: {error}"
    }
  },
  settings: {
    en: {
      title: "System Settings",
      subtitle: "Customize your experience with language and feedback options",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      feedbackTitle: "Feedback",
      feedbackPlaceholder: "Type your feedback here...",
      feedbackStatus: "Submit your feedback to help us improve",
      feedbackSuccess: "Feedback submitted successfully!",
      feedbackError: "Failed to submit feedback: {error}",
      historyTitle: "Clear History",
      historyStatus: "Clear all translation history",
      historySuccess: "Translation history cleared successfully!",
      historyError: "Failed to clear history: {error}",
      submitFeedback: "Submit Feedback",
      clearHistory: "Clear All History"
    },
    ms: {
      title: "Tetapan Sistem",
      subtitle: "Sesuaikan pengalaman anda dengan pilihan bahasa dan maklum balas",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      feedbackTitle: "Maklum Balas",
      feedbackPlaceholder: "Taip maklum balas anda di sini...",
      feedbackStatus: "Hantar maklum balas anda untuk membantu kami memperbaiki",
      feedbackSuccess: "Maklum balas berjaya dihantar!",
      feedbackError: "Gagal menghantar maklum balas: {error}",
      historyTitle: "Kosongkan Sejarah",
      historyStatus: "Kosongkan semua sejarah terjemahan",
      historySuccess: "Sejarah terjemahan berjaya dikosongkan!",
      historyError: "Gagal mengosongkan sejarah: {error}",
      submitFeedback: "Hantar Maklum Balas",
      clearHistory: "Kosongkan Semua Sejarah"
    }
  }
};

// 通用 applyLanguage 函數，根據頁面名稱和語言更新 UI
function applyLanguage(page, lang) {
  currentLang = lang;
  localStorage.setItem('language', lang);
  const translations = TRANSLATIONS[page][lang];

  // 更新通用元素（假設所有頁面有 h1 和 header p）
  const titleElement = document.querySelector('h1');
  const subtitleElement = document.querySelector('header p');
  if (titleElement) titleElement.textContent = translations.title;
  if (subtitleElement) subtitleElement.textContent = translations.subtitle;

  // 頁面特定更新
  switch (page) {
    case 'index':
      const navLinks = document.querySelectorAll('.nav-link');
      if (navLinks.length >= 5) {
        navLinks[0].textContent = translations.liveTranslationLink;
        navLinks[1].textContent = translations.roomModeLink;
        navLinks[2].textContent = translations.speechToTextLink;
        navLinks[3].textContent = translations.historyLink;
        navLinks[4].textContent = translations.settingsLink;
      }
      const welcomeElement = document.querySelector('.welcome');
      if (welcomeElement) welcomeElement.textContent = translations.welcome;
      break;

    case 'liveTranslation':
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      const uploadBtn = document.getElementById('uploadBtn');
      const result = document.getElementById('result');
      if (startBtn) startBtn.querySelector('span').textContent = translations.startRecording;
      if (stopBtn) stopBtn.querySelector('span').textContent = translations.stopRecording;
      if (uploadBtn) uploadBtn.querySelector('span').textContent = translations.transcribe;
      if (result) result.textContent = translations.statusDefault;
      break;

    case 'roomMode':
      const joinRoomBtn = document.getElementById('joinRoomBtn');
      const sendMessageBtn = document.getElementById('sendMessageBtn');
      const modeSignBtn = document.getElementById('modeSignBtn');
      const modeSpeechBtn = document.getElementById('modeSpeechBtn');
      const status = document.getElementById('status');
      if (joinRoomBtn) joinRoomBtn.textContent = translations.joinRoom;
      if (sendMessageBtn) sendMessageBtn.textContent = translations.sendMessage;
      if (modeSignBtn) modeSignBtn.textContent = translations.modeSign;
      if (modeSpeechBtn) modeSpeechBtn.textContent = translations.modeSpeech;
      if (status) status.textContent = translations.statusDisconnected;
      break;

    case 'speechToText':
      const startSpeechBtn = document.getElementById('startBtn');
      const stopSpeechBtn = document.getElementById('stopBtn');
      const uploadSpeechBtn = document.getElementById('uploadBtn');
      const resultSpeech = document.getElementById('result');
      if (startSpeechBtn) startSpeechBtn.querySelector('span').textContent = translations.startRecording;
      if (stopSpeechBtn) stopSpeechBtn.querySelector('span').textContent = translations.stopRecording;
      if (uploadSpeechBtn) uploadSpeechBtn.querySelector('span').textContent = translations.transcribe;
      if (resultSpeech) resultSpeech.textContent = translations.statusDefault;
      break;

    case 'history':
      const historyTitle = document.querySelectorAll('h2')[0];
      const noRecords = document.querySelector('.no-records');
      const deleteButtons = document.querySelectorAll('.delete-btn');
      if (historyTitle) historyTitle.textContent = translations.title;
      if (noRecords) noRecords.textContent = translations.noRecords;
      deleteButtons.forEach(btn => btn.textContent = translations.deleteRecord);
      break;

    case 'settings':
      const languageTitle = document.querySelectorAll('h2')[0];
      const feedbackTitle = document.querySelectorAll('h2')[1];
      const historyTitleSettings = document.querySelectorAll('h2')[2];
      const feedbackInput = document.getElementById('feedbackInput');
      const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
      const clearHistoryBtn = document.getElementById('clearHistoryBtn');
      const languageStatus = document.getElementById('languageStatus');
      const feedbackStatus = document.getElementById('feedbackStatus');
      const historyStatus = document.getElementById('historyStatus');
      if (languageTitle) languageTitle.textContent = translations.languageTitle;
      if (feedbackTitle) feedbackTitle.textContent = translations.feedbackTitle;
      if (historyTitleSettings) historyTitleSettings.textContent = translations.historyTitle;
      if (feedbackInput) feedbackInput.placeholder = translations.feedbackPlaceholder;
      if (submitFeedbackBtn) submitFeedbackBtn.querySelector('span').textContent = translations.submitFeedback;
      if (clearHistoryBtn) clearHistoryBtn.querySelector('span').textContent = translations.clearHistory;
      if (languageStatus) languageStatus.textContent = translations.languageStatusSuccess.replace('{lang}', lang === 'en' ? 'English' : 'Bahasa Malaysia');
      if (feedbackStatus) feedbackStatus.textContent = translations.feedbackStatus;
      if (historyStatus) historyStatus.textContent = translations.historyStatus;
      break;
  }
}
```
