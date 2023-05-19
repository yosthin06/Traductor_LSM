## Traductor_LSM
# Introducción

Este es un prototipo de traductor de Lengua de Señas Mexicana que corre en una página web para el sector médico y la gente con discapacidad auditiva pueda decir cual es su malestar.

Instalacion

Para hacer uso de este código primero tienes que clonar el repositorio en tu ordenador con el siguiente comando:

git clone https://github.com/santiagogarzam/PrototipoTraductorLSMv1.git

Una vez descargado el repositorio instala los requerimientos con el siguiente comando:

pip install -r requirements.txt

Luego, dirigete a la ubicación donde se guardó el repositorio, entra en la carpeta flask_LSM y corre el codigo de Python flask_app.py.

Copia la liga que te aparece en la línea de comandos, pegala en tu navegador web y otorga el permiso del uso de la camara.

Para instalacion de tensorflow para macbook m1 es diferente para que se revise los comandos necesarios.

Carpetas

Se tienen las siguiente carpetas

    static
    templates

Dentro de "static" se tiene el logo de la Universidad de Monterrey o se pueden guardar archivos que no tienen cambio.

Dentro de "templates" esta el archivo html donde esta la estructura de como se ve la pagina web.
Base de datos

Para tener acceso a la base de datos se tiene que solicitar y llenar un formato, el cual, se asegura que se tendra un uso academico y se puede solicitar en el siguiente correo

antonio.martinez@udem.edu 

Al obtener acceso se tendra uso del abecedario y algunas palabras clave enfocadas al sector medico como cabeza, garganta, estomago, gripe, dolor, etc.
