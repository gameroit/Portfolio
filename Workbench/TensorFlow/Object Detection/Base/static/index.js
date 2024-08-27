// BEGINNING
// Load class names
function loadClasses() {
    fetch('/classes')
        .then(response => response.json())
        .then(categories => {
            const list = document.getElementById('classes-list');
            list.innerHTML = '';
            categories.forEach(category => {
                const listItem = document.createElement('li');
                listItem.textContent = category;
                list.appendChild(listItem);
            });
        })
        .catch(error => console.error('Error loading the classes:', error));
} 
document.addEventListener('DOMContentLoaded', loadClasses);


// Access to webcam
const video = document.getElementById('video');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error('Error accessing webcam: ', err);
    });


// WHILE
// Send images
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');

function realTimeSending() {
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataUrl = canvas.toDataURL('image/jpeg');

    const formData = new FormData();
    formData.append('image', dataURLToBlob(imageDataUrl), 'image.jpg'); 

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        drawBoundingBox(data.bounding_box);
    })
    .catch(err => {
        console.error('Error during prediction: ', err);
    });
};

function dataURLToBlob(dataURL) {
    const [header, data] = dataURL.split(',');
    const mime = header.match(/:(.*?);/)[1];
    const binary = atob(data);
    const array = [];
    for (let i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    return new Blob([new Uint8Array(array)], { type: mime });
}

function drawBoundingBox(box) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    context.beginPath();
    context.rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
    context.lineWidth = 10;
    context.strokeStyle = 'green';
    context.stroke();
}
const interval = setInterval(realTimeSending, 50);