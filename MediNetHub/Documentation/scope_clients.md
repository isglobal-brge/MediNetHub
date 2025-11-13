El objetivo de este documento es definir el scope de los clientes que se van a utilizar para el desarrollo de la aplicación. 

El clients/* son los clientes que estarán instalados en los hospitales. Actualmente el cliente se encuentra en una etapa muy temprana de desarrollo, por lo que no se ha definido el scope de los clientes. La api no exige ni validación de usuario ni nada. Sin embargo el objetivo final antes de su salida a producción son las siguientes características:

* Un superadmin tendrá acceso a un dashboard de administración del client de su hospital.
* El superadmin será el encargado de otorgar los usuarios y credenciales de acceso a los investigadores que quieran conectarse a dicho cliente.
* Habrá un panel de creación de usuario y credenciales de acceso a los investigadores donde se indicará a que datasets puede acceder cada investigador. 
* En un futuro se organizarán por proyectos por lo que cualquier usuario añadido a dicho proyecto tendrá acceso a los datasets de dicho proyecto.
* Habrá un panel de administración de datasets donde se podrán añadir, editar y eliminar datasets.
* Habrá un panel de administración de usuarios donde se podrán añadir, editar y eliminar usuarios.
* Habrá un panel de administración de proyectos donde se podrán añadir, editar y eliminar proyectos.
* Habrá un panel de administración de roles donde se podrán añadir, editar y eliminar roles.
* Habrá un panel de administración de permisos donde se podrán añadir, editar y eliminar permisos.
* Habrá un panel de administración de notificaciones donde se podrán añadir, editar y eliminar notificaciones.
* Habrá un panel de administración de alertas donde se podrán añadir, editar y eliminar alertas.
* Habrá un sistema de logs para la monitorizacion de la aplicación y un usuaripo de data owner para la monitorizacion de los datasets y gestión de los mismos para cumplir con la normativa de la UE.
* Los usuarios que se le otorguen los permisos no tendrán níngu dashboard y solo acceso a la api.