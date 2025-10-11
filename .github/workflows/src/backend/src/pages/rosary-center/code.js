import { triggerCETPSim } from 'backend/chatbotBackend.jsw';

$w.onReady(() => {
  triggerCETPSim({ simType: 'cop30' });  // Auto-Start bei Load
  // HTML-Bot ansprechen
  if (window.sendMessage) {
    setTimeout(() => sendMessage('Sim läuft – Quest: COP30-Migration!'), 1000);
  }
})import { triggerCETPsin } from 'backend/chatbotBackend';

$w.onReady(async () => {
    // Setze Input oder Property für COP30 (passe an dein Element an)
    $w('#CETPsin').value = 'COP30';  // Default-Wert auf COP30 setzen (falls es ein Input ist)

    // Auto-Start: Rufe die COP30-Simulation auf Load auf
    try {
        const prompt = 'Simuliere Szenarien für die COP30 Klimakonferenz in Brasilien 2025!';  // Angepasster Prompt
        const result = await triggerCETPsin(prompt);
        
        // Handle das Ergebnis, z.B. in Chatbot oder UI anzeigen
        if (window.sendMessage) {
            window.sendMessage(result);  // Für deinen HTML-Chatbot
        } else {
            console.log('xAI-Ergebnis zu COP30:', result);  // Oder $w('#outputText').text = result;
        }
    } catch (error) {
        console.error('Fehler bei COP30-Simulation:', error);
    }
});
$w('#sendButton').onClick(async () => {
    const userPrompt = $w('#CETPsin').value + ' für COP30 Klimakonferenz';  // Ergänze mit COP30
    const result = await triggerCETPsin(userPrompt);
    // Handle result...
});