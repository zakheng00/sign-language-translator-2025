
// Define currentLang globally
window.currentLang = localStorage.getItem('language') || 'en';

// Communal TRANSLATIONS object containing translations for all pages
export const TRANSLATIONS = {
  index: {
    en: {
      title: "Sign Language Translator",
      subtitle: "Real-time translation for sign language and speech",
      welcome: "Welcome to the Sign Language Translation System",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings",
      indexLink: "Home"
    },
    ms: {
      title: "Penterjemah Bahasa Isyarat",
      subtitle: "Terjemahan masa nyata untuk bahasa isyarat dan pertuturan",
      welcome: "Selamat Datang ke Sistem Penterjemahan Bahasa Isyarat",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan",
      indexLink: "Laman Utama"
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
      feedbackStatusLoading: "Submitting feedback...",
      feedbackSuccess: "Feedback submitted successfully!",
      feedbackError: "Failed to submit feedback: {error}",
      historyTitle: "Clear History",
      historyStatus: "Clear all translation history",
      historyStatusLoading: "Clearing history...",
      historySuccess: "Translation history cleared successfully!",
      historyError: "Failed to clear history: {error}",
      submitFeedback: "Submit Feedback",
      clearHistory: "Clear All History",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
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
      feedbackStatusLoading: "Menghantar maklum balas...",
      feedbackSuccess: "Maklum balas berjaya dihantar!",
      feedbackError: "Gagal menghantar maklum balas: {error}",
      historyTitle: "Kosongkan Sejarah",
      historyStatus: "Kosongkan semua sejarah terjemahan",
      historyStatusLoading: "Mengosongkan sejarah...",
      historySuccess: "Sejarah terjemahan berjaya dikosongkan!",
      historyError: "Gagal mengosongkan sejarah: {error}",
      submitFeedback: "Hantar Maklum Balas",
      clearHistory: "Kosongkan Semua Sejarah",
      indexLink: "Laman Utama",
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
      subtitle: "Translate sign language and speech in real-time",
      startButton: "Start Translation",
      stopButton: "Stop Translation",
      status: "Ready to translate",
      statusStreaming: "Streaming translation...",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
    },
    ms: {
      title: "Terjemahan Langsung",
      subtitle: "Terjemah bahasa isyarat dan pertuturan secara masa nyata",
      startButton: "Mulakan Terjemahan",
      stopButton: "Hentikan Terjemahan",
      status: "Sedia untuk menterjemah",
      statusStreaming: "Menstrim terjemahan...",
      indexLink: "Laman Utama",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan"
    }
  },
  roomMode: {
    en: {
      title: "Room Mode",
      subtitle: "Collaborative translation in a shared room",
      joinRoom: "Join Room",
      leaveRoom: "Leave Room",
      roomStatus: "Enter a room ID to join",
      roomStatusConnected: "Connected to room: {roomId}",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
    },
    ms: {
      title: "Mod Bilik",
      subtitle: "Terjemahan kolaboratif dalam bilik berkongsi",
      joinRoom: "Sertai Bilik",
      leaveRoom: "Tinggalkan Bilik",
      roomStatus: "Masukkan ID bilik untuk menyertai",
      roomStatusConnected: "Bersambung ke bilik: {roomId}",
      indexLink: "Laman Utama",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan"
    }
  },
  speechToText: {
    en: {
      title: "Speech to Text",
      subtitle: "Convert your speech to text instantly",
      startRecording: "Start Recording",
      stopRecording: "Stop Recording",
      status: "Ready to record",
      statusRecording: "Recording...",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
    },
    ms: {
      title: "Pertuturan ke Teks",
      subtitle: "Tukar pertuturan anda kepada teks dengan segera",
      startRecording: "Mulakan Rakaman",
      stopRecording: "Hentikan Rakaman",
      status: "Sedia untuk merakam",
      statusRecording: "Sedang merakam...",
      indexLink: "Laman Utama",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan"
    }
  },
  history: {
    en: {
      title: "Translation History",
      subtitle: "View your past translations",
      noHistory: "No translation history available",
      clearHistory: "Clear All History",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings"
    },
    ms: {
      title: "Sejarah Terjemahan",
      subtitle: "Lihat terjemahan anda yang lalu",
      noHistory: "Tiada sejarah terjemahan tersedia",
      clearHistory: "Kosongkan Semua Sejarah",
      indexLink: "Laman Utama",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan"
    }
  }
};

// Universal applyLanguage function to update UI based on page and language
export function applyLanguage(page, lang) {
  window.currentLang = lang;
  localStorage.setItem('language', lang);
  const translations = TRANSLATIONS[page][lang];

  // Update common elements
  const titleElement = document.querySelector('h1');
  const subtitleElement = document.querySelector('header p');
  if (titleElement) titleElement.textContent = page === 'settings' ? `⚙️ ${translations.title}` : translations.title;
  if (subtitleElement) subtitleElement.textContent = translations.subtitle;

  // Update navigation links
  const navLinks = document.querySelectorAll('.nav-link');
  if (navLinks.length >= 5) {
    navLinks[0].textContent = translations.indexLink;
    navLinks[1].textContent = translations.liveTranslationLink;
    navLinks[2].textContent = translations.roomModeLink;
    navLinks[3].textContent = translations.speechToTextLink;
    navLinks[4].textContent = translations.historyLink;
  }

  // Page-specific updates
  if (page === 'settings') {
    const languageTitle = document.querySelectorAll('h2')[0];
    const feedbackTitle = document.querySelectorAll('h2')[1];
    const historyTitle = document.querySelectorAll('h2')[2];
    const feedbackInput = document.getElementById('feedbackInput');
    const submitFeedbackBtn = document.getElementById('submitFeedbackBtn');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const languageStatus = document.getElementById('languageStatus');
    const feedbackStatus = document.getElementById('feedbackStatus');
    const historyStatus = document.getElementById('historyStatus');
    if (languageTitle) languageTitle.textContent = translations.languageTitle;
    if (feedbackTitle) feedbackTitle.textContent = translations.feedbackTitle;
    if (historyTitle) historyTitle.textContent = translations.historyTitle;
    if (feedbackInput) feedbackInput.placeholder = translations.feedbackPlaceholder;
    if (submitFeedbackBtn) submitFeedbackBtn.querySelector('span').textContent = translations.submitFeedback;
    if (clearHistoryBtn) clearHistoryBtn.querySelector('span').textContent = translations.clearHistory;
    if (languageStatus) languageStatus.innerHTML = `
      <div class="flex items-center">
        <span class="mr-2">✅</span>
        <span>${translations.languageStatusSuccess.replace('{lang}', lang === 'en' ? 'English' : 'Bahasa Malaysia')}</span>
      </div>`;
    if (feedbackStatus) feedbackStatus.innerHTML = `
      <div class="flex items-center">
        <span>${translations.feedbackStatus}</span>
      </div>`;
    if (historyStatus) historyStatus.innerHTML = `
      <div class="flex items-center">
        <span>${translations.historyStatus}</span>
      </div>`;
  } else if (page === 'liveTranslation') {
    const startButton = document.getElementById('startTranslationBtn');
    const stopButton = document.getElementById('stopTranslationBtn');
    const status = document.getElementById('translationStatus');
    if (startButton) startButton.querySelector('span').textContent = translations.startButton;
    if (stopButton) stopButton.querySelector('span').textContent = translations.stopButton;
    if (status) status.innerHTML = `<div class="flex items-center"><span>${translations.status}</span></div>`;
  } else if (page === 'roomMode') {
    const joinRoomBtn = document.getElementById('joinRoomBtn');
    const leaveRoomBtn = document.getElementById('leaveRoomBtn');
    const roomStatus = document.getElementById('roomStatus');
    if (joinRoomBtn) joinRoomBtn.querySelector('span').textContent = translations.joinRoom;
    if (leaveRoomBtn) leaveRoomBtn.querySelector('span').textContent = translations.leaveRoom;
    if (roomStatus) roomStatus.innerHTML = `<div class="flex items-center"><span>${translations.roomStatus}</span></div>`;
  } else if (page === 'speechToText') {
    const startRecordingBtn = document.getElementById('startRecordingBtn');
    const stopRecordingBtn = document.getElementById('stopRecordingBtn');
    const status = document.getElementById('recordingStatus');
    if (startRecordingBtn) startRecordingBtn.querySelector('span').textContent = translations.startRecording;
    if (stopRecordingBtn) stopRecordingBtn.querySelector('span').textContent = translations.stopRecording;
    if (status) status.innerHTML = `<div class="flex items-center"><span>${translations.status}</span></div>`;
  } else if (page === 'history') {
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const historyList = document.getElementById('historyList');
    if (clearHistoryBtn) clearHistoryBtn.querySelector('span').textContent = translations.clearHistory;
    if (historyList && historyList.children.length === 0) {
      historyList.innerHTML = `<div class="flex items-center"><span>${translations.noHistory}</span></div>`;
    }
  }
}
