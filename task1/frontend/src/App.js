import React, { useState, useEffect, useRef } from "react";
import "./App.css";
 // Import the splash screen styles
import SplashScreen from "./SplashScreen";

function TypingIndicator() {
  return (
    <div className="typing-indicator" aria-label="Bot is typing">
      <span></span><span></span><span></span>
    </div>
  );
}

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [typing, setTyping] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    const savedTheme = localStorage.getItem("darkMode");
    return savedTheme === null ? true : JSON.parse(savedTheme);
  });
  const [botText, setBotText] = useState("");
  const botFullTextRef = useRef("");
  const typingIntervalRef = useRef(null);
  const messagesEndRef = useRef(null);
  const [showSplash, setShowSplash] = useState(true);

  // Handle splash screen fade-out
  useEffect(() => {
    const timeout = setTimeout(() => {
      const splashEl = document.querySelector(".splash-screen");
      if (splashEl) splashEl.classList.add("hide");

      setTimeout(() => {
        setShowSplash(false);
      }, 800);
    }, 2000);

    return () => clearTimeout(timeout);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, typing, botText]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: input,
      sender: "user",
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setTyping(true);
    setBotText("");

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });
      const data = await response.json();
      botFullTextRef.current = data.response || "Sorry, no response.";

      let index = 0;
      clearInterval(typingIntervalRef.current);
      typingIntervalRef.current = setInterval(() => {
        index++;
        setBotText(botFullTextRef.current.slice(0, index));
        if (index === botFullTextRef.current.length) {
          clearInterval(typingIntervalRef.current);
          const botMessage = {
            id: Date.now() + 1,
            text: botFullTextRef.current,
            sender: "bot",
            time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          };
          setMessages((prev) => [...prev, botMessage]);
          setTyping(false);
          setBotText("");
        }
      }, 30);
    } catch (error) {
      console.error("Error:", error);
      setTyping(false);
    }
  };

  return (
    <>
      {showSplash && <SplashScreen />}

      <div className={`app ${darkMode ? "dark" : "light"}`}>
        <header className="app-header">
          <h1>CodeSoft College Chatbot</h1>
          <button
            className="theme-toggle"
            aria-label="Toggle Dark Mode"
            onClick={() => {
              setDarkMode((prev) => {
                localStorage.setItem("darkMode", JSON.stringify(!prev));
                return !prev;
              });
            }}
          >
            {darkMode ? "☀️ Light" : "🌙 Dark"}
          </button>
        </header>

        <main className="chat-container" role="main" aria-live="polite">
          <div className="messages">
            {messages.map(({ id, text, sender, time }) => (
              <div key={id} className={`message ${sender}`} tabIndex={0}>
                <p className="message-text">{text}</p>
                <span className="timestamp" aria-label={`Sent at ${time}`}>
                  {time}
                </span>
              </div>
            ))}

            {typing && botText !== "" && (
              <div className="message bot typing" aria-live="assertive" aria-atomic="true">
                <p className="message-text">{botText}</p>
                <TypingIndicator />
              </div>
            )}

            {typing && botText === "" && (
              <div className="message bot typing" aria-live="assertive" aria-atomic="true">
                <TypingIndicator />
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form className="input-area" onSubmit={handleSubmit} aria-label="Send message form">
            <input
              type="text"
              aria-label="Type your message"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message here..."
              autoComplete="off"
              spellCheck="false"
            />
            <button
              type="submit"
              className="send-button"
              onClick={(e) => {
                e.currentTarget.classList.add("clicked");
                setTimeout(() => {
                  e.currentTarget?.classList.remove("clicked");
                }, 300);
              }}
              aria-label="Send message"
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="white" viewBox="0 0 24 24" width="24" height="24">
                <path d="M2 21l21-9L2 3v7l15 2-15 2z" />
              </svg>
            </button>
          </form>
        </main>
      </div>
    </>
  );
}

export default App;
