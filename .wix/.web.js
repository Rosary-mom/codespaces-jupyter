<iframe style="border:none; width:100%; height:600px;" srcdoc='
  <!DOCTYPE html>
  <html lang="de">
  <head>
    <meta charset="UTF-8">
    <title>Grok Chatbot</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 0; padding: 10px; background: #f0f0f0; }
      #chatBox { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background: white; }
      #inputBox { width: calc(100% - 80px); padding: 5px; }
      #sendButton { padding: 5px 10px; }
      #modelSelect { margin-left: 10px; padding: 5px; }
    </style>
  </head>
  <body>
    <h2>Intelligenter Grok-Chatbot für COP30-Szenarien</h2>
    <div id="chatBox"></div>
    <input type="text" id="inputBox" placeholder="z.B. Simuliere optimistisches Szenario...">
    <select id="modelSelect">
      <option value="grok-4">Expert (grok-4)</option>
      <option value="grok-4-fast-reasoning">Fast Beta (grok-4-fast-reasoning)</option>
    </select>
    <button id="sendButton">Senden</button>

    <script>
      const apiKey = "DEIN_SECRET_API_KEY_HIER";  // Ersetze hier!
      let chatHistory = [];

      document.getElementById("sendButton").addEventListener("click", async () => {
        const prompt = document.getElementById("inputBox").value;
        const model = document.getElementById("modelSelect").value;
        if (!prompt) return;

        const chatBox = document.getElementById("chatBox");
        chatBox.innerHTML += `<p><strong>Du:</strong> ${prompt}</p>`;
        document.getElementById("inputBox").value = "";

        try {
          console.log("Sending request with model:", model);  // Debug-Log
          const messages = [
            { role: "system", content: "Du bist ein KI-Experte für COP30-Szenarien (Klimakonferenz 2025 in Brasilien). Simuliere optimistische, realistische oder pessimistische Szenarien zu Themen wie Finanzierung, NDCs, Amazonas-Schutz. Sei fundiert und interaktiv." },
            ...chatHistory,
            { role: "user", content: prompt }
          ];

          const response = await fetch("https://api.x.ai/v1/chat/completions", {
            method: "POST",
            headers: { "Authorization": `Bearer ${apiKey}`, "Content-Type": "application/json" },
            body: JSON.stringify({ model: model, messages: messages, max_tokens: 1000 })
          });

          console.log("API Status:", response.status);  // Debug-Log

          if (!response.ok) throw new Error("API-Fehler: " + response.status);

          const data = await response.json();
          const result = data.choices[0].message.content;

          chatBox.innerHTML += `<p><strong>Bot:</strong> ${result}</p>`;
          chatHistory.push({ role: "user", content: prompt }, { role: "assistant", content: result });
          chatBox.scrollTop = chatBox.scrollHeight;
        } catch (error) {
          console.error("Error:", error);  // Debug in Console
          chatBox.innerHTML += `<p><strong>Fehler:</strong> ${error.message}</p>`;
        }
      });
    </script>
  </body>
  </html>
'></iframe>