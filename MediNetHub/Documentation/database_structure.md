# Estructura de la Base de Dades

## Visió General
Aquest document descriu l'estructura de la base de dades utilitzada en el projecte MediNet. El sistema utilitza Django ORM amb una base de dades SQLite per defecte, tot i que és compatible amb altres sistemes de gestió de bases de dades com PostgreSQL.

## Models

### Project
Model per organitzar les connexions hospitalàries i datasets.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | ForeignKey | Usuari propietari del projecte |
| name | CharField | Nom del projecte (màxim 100 caràcters) |
| description | TextField | Descripció del projecte (opcional) |
| color | CharField | Color hexadecimal per la interfície (opcions predefinides) |
| created_at | DateTimeField | Data de creació del projecte |
| updated_at | DateTimeField | Data d'última actualització |

**Relacions**: 
- Constraint unique_together entre user i name
- Ordenació per nom per defecte

### UserProfile
Extensió del model d'usuari de Django per afegir informació addicional.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | OneToOneField | Relació amb el model User de Django |
| organization | CharField | Organització a la qual pertany l'usuari (opcional) |
| bio | TextField | Biografia o descripció de l'usuari (opcional) |
| created_at | DateTimeField | Data de creació del perfil |
| updated_at | DateTimeField | Data d'última actualització |

### ModelConfig
Emmagatzema les configuracions de xarxes neuronals.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | ForeignKey | Usuari que ha creat la configuració |
| name | CharField | Nom de la configuració (màxim 100 caràcters) |
| description | TextField | Descripció detallada (opcional) |
| framework | CharField | Framework utilitzat (per defecte 'pt' per PyTorch) |
| config_json | JSONField | Configuració en format JSON |
| created_at | DateTimeField | Data de creació |
| updated_at | DateTimeField | Data d'última actualització |

### TrainingJob
Gestiona els treballs d'entrenament.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | ForeignKey | Usuari que ha iniciat l'entrenament |
| model_config | ForeignKey | Configuració del model utilitzat |
| name | CharField | Nom del treball (màxim 100 caràcters) |
| description | TextField | Descripció del treball (opcional) |
| status | CharField | Estat actual (pending/server_ready/running/completed/failed/cancelled) |
| dataset_id | CharField | ID del dataset (camp llegat, opcional) |
| dataset_ids | JSONField | IDs dels datasets utilitzats (suporta múltiples datasets) |
| metrics_file | CharField | Ruta al fitxer de mètriques (opcional) |
| metrics_json | JSONField | Mètriques d'entrenament en format JSON |
| model_file_path | CharField | Ruta al fitxer del model entrenat (opcional) |
| config_json | JSONField | Configuració addicional en format JSON |
| progress | IntegerField | Progrés de l'entrenament (0-100) |
| current_round | IntegerField | Ronda actual d'entrenament |
| total_rounds | IntegerField | Total de rondes a completar |
| started_at | DateTimeField | Data i hora d'inici (opcional) |
| completed_at | DateTimeField | Data i hora de finalització (opcional) |
| created_at | DateTimeField | Data de creació |
| updated_at | DateTimeField | Data d'última actualització |
| clients_status | JSONField | Estat dels clients en format JSON |
| logs | TextField | Registres d'entrenament (opcional) |
| server_pid | IntegerField | PID del procés del servidor Flower (opcional) |
| training_duration | FloatField | Durada total de l'entrenament en segons (opcional) |

### Connection
Gestiona les connexions per a l'aprenentatge federat.

| Camp | Tipus | Descripció |
|------|-------|------------|
| name | CharField | Nom de la connexió (màxim 100 caràcters) |
| ip | CharField | Adreça IP (suporta IPv6, màxim 45 caràcters) |
| port | IntegerField | Port de connexió |
| username | CharField | Nom d'usuari (opcional, màxim 100 caràcters) |
| password | CharField | Contrasenya (opcional, màxim 100 caràcters) |
| active | BooleanField | Estat de la connexió (per defecte True) |
| user | ForeignKey | Usuari propietari |
| project | ForeignKey | Projecte associat (opcional) |
| created_at | DateTimeField | Data de creació |
| updated_at | DateTimeField | Data d'última actualització |

### Dataset
Informació sobre els conjunts de dades.

| Camp | Tipus | Descripció |
|------|-------|------------|
| connection | ForeignKey | Connexió associada |
| dataset_name | CharField | Nom del dataset (màxim 255 caràcters) |
| class_label | CharField | Etiqueta de classe (màxim 255 caràcters) |
| num_columns | IntegerField | Nombre de columnes (per defecte 0) |
| num_rows | IntegerField | Nombre de files (per defecte 0) |
| size | IntegerField | Mida del dataset (per defecte 0) |
| created_at | DateTimeField | Data de creació |
| updated_at | DateTimeField | Data d'última actualització |

### Model
Emmagatzema els models entrenats.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | ForeignKey | Usuari propietari |
| name | CharField | Nom del model (màxim 100 caràcters) |
| config | TextField | Configuració del model |
| created_at | DateTimeField | Data de creació |
| updated_at | DateTimeField | Data d'última actualització |

### Notification
Sistema de notificacions.

| Camp | Tipus | Descripció |
|------|-------|------------|
| user | ForeignKey | Usuari destinatari |
| title | CharField | Títol de la notificació (màxim 100 caràcters) |
| message | TextField | Missatge |
| link | CharField | Enllaç opcional (màxim 200 caràcters) |
| is_read | BooleanField | Estat de lectura (per defecte False) |
| created_at | DateTimeField | Data de creació |

**Relacions**: Ordenació per data de creació descendent per defecte

## Relacions Principals
- **UserProfile** té una relació one-to-one amb User
- **Project** té una relació many-to-one amb User
- **ModelConfig** té una relació many-to-one amb User
- **TrainingJob** té relacions many-to-one amb User i ModelConfig
- **Connection** té relacions many-to-one amb User i Project (opcional)
- **Dataset** té una relació many-to-one amb Connection
- **Model** té una relació many-to-one amb User
- **Notification** té una relació many-to-one amb User

## Constraints i Validacions
- **Project**: Combinació única de user i name
- **Connection**: Suport per IPv6 amb longitud màxima de 45 caràcters per IP
- **TrainingJob**: Estats predefinits amb choices
- **Notification**: Ordenació automàtica per data de creació descendent

## Consideracions Especials
- Els camps JSONField permeten emmagatzemar estructures de dades complexes
- Els camps opcionals tenen blank=True i null=True quan és apropiat
- Les dates de creació i actualització s'estableixen automàticament
- El model Project inclou opcions de color predefinides per la interfície
- El model TrainingJob suporta tant dataset individual com múltiples datasets per compatibilitat 