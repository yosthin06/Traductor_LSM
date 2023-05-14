// Obtener el elemento de video y el botón de tomar foto
const video = document.getElementById('video');
const boton = document.getElementById('boton');

// Solicitar permiso al usuario para acceder a la cámara
navigator.mediaDevices.getUserMedia({video: true})
  .then(function(stream) {
    // Mostrar la vista previa de la cámara en el elemento de video
    video.srcObject = stream;
  })
  .catch(function(error) {
    console.log('Error al obtener el acceso a la cámara: ' + error);
  });

// Agregar un evento de clic al botón de tomar foto
boton.addEventListener('click', function() {
	// Crear un canvas a partir del elemento de video
	const canvas = document.createElement('canvas');
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	const ctx = canvas.getContext('2d');
	ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

	// Obtener la imagen en formato base64
	const dataURL = canvas.toDataURL('image/png');

	// Mostrar la imagen en la página
	const imagen = new Image();
	imagen.src = dataURL;
	document.body.appendChild(imagen);
});