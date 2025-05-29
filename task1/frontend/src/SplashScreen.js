// SplashScreen.js
import React from "react";
import "./App.css";

function SplashScreen() {
  return (
    <div className="splash-screen">
      <div className="logo-container">
        <div className="logo-icon">🎓</div>
        <h1 className="logo-text">CodeSoft College Chatbot</h1>
      </div>
      <div className="loader" aria-label="Loading" />
      <p>Initializing AI Assistant...</p>
    </div>
  );
}

export default SplashScreen;
