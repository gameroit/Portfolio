const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Line params
ctx.lineWidth = 15
ctx.lineCap = 'round'
ctx.strokeStyle ='#fff'

// BG params
ctx.fillStyle = 'black'; 
ctx.fillRect(0, 0, canvas.width, canvas.height); 


// Start drawing
canvas.addEventListener('mousedown', () => {
    isDrawing = true;
    ctx.beginPath();
});

// Stop drawing
canvas.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Draw
canvas.addEventListener('mousemove', (event) => {
    if (isDrawing) {
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        ctx.lineTo(x, y);
        ctx.stroke();
    }
});

// Send image
document.getElementById('sendButton').addEventListener('click', () => {
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('file', blob, 'drawing.png');

        // Call /predict
        fetch('/predict', {
            method: 'POST',
            body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('prediction').innerText = `Clase: ${data.predicted_class} (Probability: ${data.confidence.toFixed(2)})`;
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }, 'image/png');
});

// Clear image
document.getElementById('clearButton').addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillRect(0, 0, canvas.width, canvas.height); 
});