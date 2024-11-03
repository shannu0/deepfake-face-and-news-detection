document.getElementById('deepfake-form').onsubmit = function(e) {
    e.preventDefault(); // Prevent the default form submission

    const formData = new FormData(this);
    
    fetch('/predict_deepfake', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Error:', data.error);
            alert('Error: ' + data.error);
        } else {
            console.log('Predictions:', data.predictions);
            document.getElementById('result').innerText = `Real: ${data.predictions.real.toFixed(2)}, Fake: ${data.predictions.fake.toFixed(2)}`;
            
            // Display the face with mask
            const imgElement = document.getElementById('face-with-mask');
            imgElement.src = `data:image/jpeg;base64,${data.face_with_mask}`;
            imgElement.style.display = 'block'; // Show the image
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error);
    });
};
