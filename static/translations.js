// Define currentLang globally
window.currentLang = localStorage.getItem('language') || 'en';

// Communal TRANSLATIONS object containing translations for all pages
export const TRANSLATIONS = {
  index: {
    en: {
      title: "Sign Language Translator",
      subtitle: "Breaking barriers through intelligent real-time translation",
      welcome: "Welcome to the Sign Language Translation System",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
      indexLink: "Home",
      liveTranslationLink: "Live Translation",
      roomModeLink: "Room Mode",
      speechToTextLink: "Speech to Text",
      historyLink: "History",
      settingsLink: "Settings",
      liveSignTranslation: "Live Sign Translation",
      liveSignDesc: "Experience real-time sign language translation with advanced AI recognition using your webcam",
      speechToTextDesc: "Convert spoken words into accurate text with natural language processing",
      roomModeDesc: "Create collaborative spaces for seamless real-time translation with multiple users",
      historyDesc: "Access and review your complete translation records and conversation history",
      settingsDesc: "Customize language preferences and fine-tune AI model parameters"
    },
    ms: {
      title: "Penterjemah Bahasa Isyarat",
      subtitle: "Memecah halangan melalui terjemahan pintar masa nyata",
      welcome: "Selamat Datang ke Sistem Penterjemahan Bahasa Isyarat",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
      indexLink: "Laman Utama",
      liveTranslationLink: "Terjemahan Langsung",
      roomModeLink: "Mod Bilik",
      speechToTextLink: "Pertuturan ke Teks",
      historyLink: "Sejarah",
      settingsLink: "Tetapan",
      liveSignTranslation: "Terjemahan Isyarat Langsung",
      liveSignDesc: "Alami terjemahan bahasa isyarat masa nyata dengan pengecaman AI canggih menggunakan kamera web anda",
      speechToTextDesc: "Tukarkan perkataan yang dituturkan kepada teks yang tepat dengan pemprosesan bahasa semula jadi",
      roomModeDesc: "Cipta ruang kolaboratif untuk terjemahan masa nyata yang lancar dengan berbilang pengguna",
      historyDesc: "Akses dan semak rekod terjemahan lengkap dan sejarah perbualan anda",
      settingsDesc: "Sesuaikan pilihan bahasa dan laraskan parameter model AI"
    }
  },
  settings: {
    en: {
      title: "System Settings",
      subtitle: "Customize your experience with language and feedback options",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      submitFeedback: "📬 Submit Feedback",
      clearHistory: "🗑️ Clear All History",
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
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      submitFeedback: "📬 Hantar Maklum Balas",
      clearHistory: "🗑️ Kosongkan Semua Sejarah",
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
      translationTitle: "Translation Controls",
      startButton: "Start Translation",
      stopButton: "Stop Translation",
      status: "Ready to translate",
      statusStreaming: "Streaming translation...",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      translationTitle: "Kawalan Terjemahan",
      startButton: "Mulakan Terjemahan",
      stopButton: "Hentikan Terjemahan",
      status: "Sedia untuk menterjemah",
      statusStreaming: "Menstrim terjemahan...",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      roomTitle: "Room Controls",
      joinRoom: "Join Room",
      leaveRoom: "Leave Room",
      roomIdPlaceholder: "Room ID",
      roomStatus: "Enter a room ID to join",
      roomStatusConnected: "Connected to room: {roomId}",
      roomStatusError: "Failed to join room: {error}",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      roomTitle: "Kawalan Bilik",
      joinRoom: "Sertai Bilik",
      leaveRoom: "Tinggalkan Bilik",
      roomIdPlaceholder: "ID Bilik",
      roomStatus: "Masukkan ID bilik untuk menyertai",
      roomStatusConnected: "Bersambung ke bilik: {roomId}",
      roomStatusError: "Gagal menyertai bilik: {error}",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      recordingTitle: "Recording Controls",
      startRecording: "Start Recording",
      stopRecording: "Stop Recording",
      status: "Ready to record",
      statusRecording: "Recording...",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      recordingTitle: "Kawalan Rakaman",
      startRecording: "Mulakan Rakaman",
      stopRecording: "Hentikan Rakaman",
      status: "Sedia untuk merakam",
      statusRecording: "Sedang merakam...",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      historyTitle: "Translation Records",
      noHistory: "No translation history available",
      clearHistory: "Clear All History",
      languageTitle: "Language Settings",
      languageStatusLoading: "Loading language settings...",
      languageStatusSuccess: "Language updated to {lang}",
      languageStatusError: "Failed to update language: {error}",
      langEnButton: "🇬🇧 English",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
      historyTitle: "Rekod Terjemahan",
      noHistory: "Tiada sejarah terjemahan tersedia",
      clearHistory: "Kosongkan Semua Sejarah",
      languageTitle: "Tetapan Bahasa",
      languageStatusLoading: "Memuatkan tetapan bahasa...",
      languageStatusSuccess: "Bahasa ditukar ke {lang}",
      languageStatusError: "Gagal menukar bahasa: {error}",
      langEnButton: "🇬🇧 Bahasa Inggeris",
      langMsButton: "🇲🇾 Bahasa Malaysia",
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
  
  // Broadcast language change to other tabs/windows
  window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang, page: page } }));
  
  const translations = TRANSLATIONS[page][lang];
  if (!translations) return;

  // Update page title in browser tab
  document.title = `Sign Language Translation System - ${translations.title}`;

  // Update common elements
  const titleElement = document.querySelector('h1');
  const subtitleElement = document.querySelector('header p');
  
  if (titleElement) {
    titleElement.textContent = page === 'settings' ? `⚙️ ${translations.title}` : translations.title;
  }
  if (subtitleElement) {
    subtitleElement.textContent = translations.subtitle;
  }

  // Update navigation links
  const navLinks = document.querySelectorAll('nav a');
  if (navLinks.length >= 5) {
    navLinks[0].textContent = translations.indexLink;
    navLinks[1].textContent = translations.liveTranslationLink;
    navLinks[2].textContent = translations.roomModeLink;
    navLinks[3].textContent = translations.speechToTextLink;
    navLinks[4].textContent = translations.historyLink;
  }

  // Update language buttons
  const langEnBtn = document.getElementById('langEnBtn');
  const langMsBtn = document.getElementById('langMsBtn');
  if (langEnBtn) {
    const span = langEnBtn.querySelector('span');
    if (span) span.textContent = translations.langEnButton;
  }
  if (langMsBtn) {
    const span = langMsBtn.querySelector('span');
    if (span) span.textContent = translations.langMsButton;
  }

  // Page-specific updates
  updatePageSpecificElements(page, translations, lang);
}

function updatePageSpecificElements(page, translations, lang) {
  if (page === 'index') {
    // Update feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    if (featureCards.length >= 5) {
      // Live Sign Translation card
      const liveCard = featureCards[0];
      const liveTitle = liveCard.querySelector('h2');
      const liveDesc = liveCard.querySelector('p');
      if (liveTitle) liveTitle.textContent = translations.liveSignTranslation || "Live Sign Translation";
      if (liveDesc) liveDesc.textContent = translations.liveSignDesc || "Experience real-time sign language translation";

      // Speech to Text card
      const speechCard = featureCards[1];
      const speechTitle = speechCard.querySelector('h2');
      const speechDesc = speechCard.querySelector('p');
      if (speechTitle) speechTitle.textContent = translations.speechToTextLink || "Speech to Text";
      if (speechDesc) speechDesc.textContent = translations.speechToTextDesc || "Convert spoken words into text";

      // Room Mode card
      const roomCard = featureCards[2];
      const roomTitle = roomCard.querySelector('h2');
      const roomDesc = roomCard.querySelector('p');
      if (roomTitle) roomTitle.textContent = translations.roomModeLink || "Room Mode";
      if (roomDesc) roomDesc.textContent = translations.roomModeDesc || "Collaborative translation spaces";

      // History card
      const historyCard = featureCards[3];
      const historyTitle = historyCard.querySelector('h2');
      const historyDesc = historyCard.querySelector('p');
      if (historyTitle) historyTitle.textContent = translations.historyLink || "Translation History";
      if (historyDesc) historyDesc.textContent = translations.historyDesc || "Review your translation history";

      // Settings card
      const settingsCard = featureCards[4];
      const settingsTitle = settingsCard.querySelector('h2');
      const settingsDesc = settingsCard.querySelector('p');
      if (settingsTitle) settingsTitle.textContent = translations.settingsLink || "Settings";
      if (settingsDesc) settingsDesc.textContent = translations.settingsDesc || "Customize your preferences";
    }
  } else if (page === 'settings') {
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
    if (submitFeedbackBtn) {
      const span = submitFeedbackBtn.querySelector('span');
      if (span) span.textContent = translations.submitFeedback;
    }
    if (clearHistoryBtn) {
      const span = clearHistoryBtn.querySelector('span');
      if (span) span.textContent = translations.clearHistory;
    }
    
    if (languageStatus) {
      languageStatus.className = 'status-bar status-success';
      languageStatus.innerHTML = `
        <div class="flex items-center">
          <span class="mr-2">✅</span>
          <span>${translations.languageStatusSuccess.replace('{lang}', lang === 'en' ? 'English' : 'Bahasa Malaysia')}</span>
        </div>`;
    }
    if (feedbackStatus) {
      feedbackStatus.innerHTML = `
        <div class="flex items-center">
          <span>${translations.feedbackStatus}</span>
        </div>`;
    }
    if (historyStatus) {
      historyStatus.innerHTML = `
        <div class="flex items-center">
          <span>${translations.historyStatus}</span>
        </div>`;
    }
  }
  // Add more page-specific updates as needed
}

// Listen for language changes from other tabs/windows
window.addEventListener('languageChanged', (event) => {
  const { language, page } = event.detail;
  if (window.currentLang !== language) {
    // Detect current page
    const currentPage = detectCurrentPage();
    applyLanguage(currentPage, language);
  }
});

// Function to detect current page
function detectCurrentPage() {
  const path = window.location.pathname;
  if (path.includes('settings')) return 'settings';
  if (path.includes('live-translation')) return 'liveTranslation';
  if (path.includes('room-mode')) return 'roomMode';
  if (path.includes('speech-to-text')) return 'speechToText';
  if (path.includes('history')) return 'history';
  return 'index';
}

// Initialize language on page load
document.addEventListener('DOMContentLoaded', () => {
  const currentPage = detectCurrentPage();
  const currentLang = localStorage.getItem('language') || 'en';
  applyLanguage(currentPage, currentLang);
});
