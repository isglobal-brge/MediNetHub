# MediNet - Guia d'Usuari Completa

## Introducció

Aquesta guia t'ajudarà a utilitzar totes les funcionalitats de MediNet, des del registre inicial fins a l'entrenament de models d'intel·ligència artificial federats.

## 1. Registre i Autenticació

### Crear un Compte
1. Visita la pàgina principal de MediNet
2. Fes clic a "Registrar-se"
3. Omple el formulari amb:
   - Nom d'usuari
   - Correu electrònic
   - Nom i cognoms
   - Organització (hospital/institució)
   - Contrasenya segura
4. Confirma el registre

### Iniciar Sessió
- Utilitza el teu nom d'usuari i contrasenya
- El sistema et portarà al teu dashboard personalitzat

## 2. Dashboard Principal

### Vista General
El dashboard mostra:
- **Resum d'activitat**: Entrenaments recents i estadístiques
- **Notificacions**: Alertes sobre l'estat dels teus projectes
- **Accés ràpid**: Enllaços a les funcions més utilitzades
- **Projectes actius**: Llista dels teus projectes en curs

### Navegació
- **Panel**: Dashboard principal
- **Datasets**: Gestió de dades
- **Models**: Dissenyador de models
- **Training**: Entrenament federat
- **Notifications**: Centre de notificacions

## 3. Gestió de Projectes

### Crear un Nou Projecte
1. Accedeix a la gestió de projectes
2. Fes clic "Nou Projecte"
3. Omple la informació:
   - **Nom**: Identificador únic del projecte
   - **Descripció**: Objectius i context
   - **Color**: Per identificació visual
4. Guarda el projecte

### Gestionar Projectes
- **Editar**: Modifica nom, descripció o color
- **Eliminar**: Suprimeix projectes no necessaris
- **Seleccionar**: Canvia entre projectes actius

## 4. Connexions Hospitalàries

### Afegir una Nova Connexió
1. Ves a "Datasets" → "Manage Connections"
2. Fes clic "Add New Connection"
3. Configura la connexió:
   - **Nom**: Identificador de l'hospital
   - **IP**: Adreça del servidor hospitalari
   - **Port**: Port de comunicació
   - **Projecte**: Assigna a un projecte (opcional)
   - **Credencials**: Username/password si és necessari

### Validar Connexions
- **Test Connection**: Verifica que la connexió funciona
- **Status**: Monitoritza l'estat (activa/inactiva)
- **Sync Data**: Sincronitza informació dels datasets

### Gestió de Connexions
- **Editar**: Modifica paràmetres de connexió
- **Desactivar**: Pausa connexions temporalment
- **Eliminar**: Suprimeix connexions obsoletes

## 5. Gestió de Datasets

### Visualitzar Datasets
1. Selecciona una connexió activa
2. Fes clic "Fetch Datasets" per sincronitzar
3. Revisa la informació disponible:
   - Nom del dataset
   - Nombre de files i columnes
   - Etiquetes de classe
   - Mida dels fitxers

### Previsualitzar Dades
- **Preview**: Veu una mostra de les dades (sense descarregar)
- **Statistics**: Estadístiques descriptives dels camps numèrics
- **Distribution**: Distribució de variables categòriques

### Seleccionar Datasets per Entrenament
1. Marca els datasets que vols utilitzar
2. Utilitza "Select for Training" per afegir-los a la selecció
3. Revisa la selecció abans de procedir a l'entrenament

## 6. Dissenyador de Models

### Accés al Dissenyador
- Ves a "Models" → "Model Designer"
- Aquí pots crear noves arquitectures de xarxes neuronals

### Crear un Model Nou

#### Configuració Bàsica
1. **Nom del Model**: Identificador únic
2. **Descripció**: Propòsit i característiques
3. **Framework**: PyTorch (per defecte)

#### Disseny de l'Arquitectura
1. **Input Layer**: Defineix les dimensions d'entrada
2. **Hidden Layers**: Afegeix capes ocultes
   - Dense/Linear: Capes completament connectades
   - Convolutional: Per dades d'imatge
   - LSTM/GRU: Per seqüències temporals
3. **Output Layer**: Capa de sortida segons el tipus de problema
4. **Activation Functions**: ReLU, Sigmoid, Tanh, etc.

#### Configuració Avançada
- **Optimizer**: Adam, SGD, RMSprop
- **Loss Function**: CrossEntropy, MSE, BCE
- **Learning Rate**: Taxa d'aprenentatge
- **Batch Size**: Mida del lot d'entrenament
- **Epochs**: Nombre d'iteracions

### Templates Predefinits
- **Classificació Binària**: Models per diagnòstic sí/no
- **Classificació Multi-classe**: Múltiples categories
- **Regressió**: Predicció de valors continus
- **Deep Learning**: Arquitectures complexes per imatges

### Guardar i Gestionar Models
- **Save**: Guarda la configuració
- **Load**: Carrega models existents
- **Clone**: Duplica models per variants
- **Delete**: Elimina models obsolets

## 7. Entrenament Federat

### Iniciar un Entrenament

#### Preparació
1. **Selecciona un Model**: Des del dissenyador o biblioteca
2. **Tria Datasets**: Selecciona les dades per entrenar
3. **Configura Paràmetres**:
   - Nombre de rondes federats
   - Critèris de convergència
   - Paràmetres de simulació

#### Configuració de l'Entrenament
1. **Nom del Job**: Identificador de l'entrenament
2. **Descripció**: Objectius i notes
3. **Validation Split**: Percentatge per validació
4. **Early Stopping**: Criteris d'aturada anticipada

#### Llançar l'Entrenament
- Revisa tots els paràmetres
- Fes clic "Start Training"
- El sistema començarà la simulació federada

### Monitoratge en Temps Real

#### Dashboard d'Entrenament
- **Progress Bar**: Progrés actual (0-100%)
- **Current Round**: Ronda federada actual
- **Status**: pending/running/completed/failed
- **Estimated Time**: Temps restant estimat

#### Mètriques en Temps Real
- **Loss**: Evolució de la funció de pèrdua
- **Accuracy**: Precisió del model
- **Precision/Recall/F1**: Mètriques detallades
- **Learning Curves**: Gràfics de convergència

#### Estat dels Clients
- **Clients Active**: Hospitals participants
- **Client Status**: Estat individual de cada client
- **Data Distribution**: Distribució de dades per client

### Control de l'Entrenament
- **Pause**: Pausa l'entrenament
- **Resume**: Continua després d'una pausa
- **Stop**: Atura permanentment
- **Cancel**: Cancel·la l'entrenament

## 8. Anàlisi i Comparació

### Dashboard de Resultats
Després de completar un entrenament:

#### Mètriques Finals
- **Model Performance**: Rendiment final del model
- **Training History**: Historial complet d'entrenament
- **Convergence Analysis**: Anàlisi de convergència
- **Client Contributions**: Contribució de cada hospital

#### Visualitzacions
- **Loss Curves**: Evolució de la pèrdua
- **Accuracy Curves**: Millora de la precisió
- **Confusion Matrix**: Matriu de confusió
- **ROC Curves**: Corbes ROC per classificació

### Comparació de Models
1. **Select Models**: Selecciona fins a 3 models per comparar
2. **Side-by-side**: Comparació visual de mètriques
3. **Performance Metrics**: Taula comparativa detallada
4. **Statistical Tests**: Tests de significació estadística

### Exportació de Resultats
- **Download Model**: Descarrega el model entrenat
- **Export Metrics**: Exporta mètriques en CSV/JSON
- **Generate Report**: Genera informe automàtic
- **Share Results**: Comparteix amb col·laboradors

## 9. Sistema de Notificacions

### Tipus de Notificacions
- **Training Started**: Inici d'entrenament
- **Training Completed**: Finalització exitosa
- **Training Failed**: Errors durant l'entrenament
- **Data Updates**: Actualitzacions dels datasets
- **System Alerts**: Alertes del sistema

### Gestió de Notificacions
- **Mark as Read**: Marca com llegides
- **Filter**: Filtra per tipus o data
- **Archive**: Arxiva notificacions antigues
- **Settings**: Configura preferències de notificació

## 10. Perfil i Configuració

### Perfil d'Usuari
- **Personal Info**: Nom, email, organització
- **Profile Picture**: Avatar personalitzat
- **Bio**: Descripció professional
- **Contact Info**: Informació de contacte

### Configuració de Compte
- **Change Password**: Canvia la contrasenya
- **Email Preferences**: Configuració d'emails
- **Privacy Settings**: Configuració de privacitat
- **API Keys**: Claus per integracions

### Configuració de Preferències
- **Dashboard Layout**: Personalitza el dashboard
- **Notification Settings**: Tipus de notificacions
- **Theme**: Tema clar/fosc
- **Language**: Idioma de la interfície

## 11. Solució de Problemes

### Problemes Comuns

#### Connexions
- **Error de connexió**: Verifica IP i port
- **Timeout**: Augmenta el timeout de connexió
- **Credencials**: Revisa username/password

#### Entrenament
- **Entrenament fallit**: Revisa logs per errors
- **Baixa precisió**: Ajusta hiperparàmetres
- **Lentitud**: Redueix batch size o model complexity

#### Dades
- **Datasets no apareixen**: Sincronitza connexions
- **Errors de format**: Verifica format de dades
- **Dades insuficients**: Assegura't de tenir prou mostres

### Suport Tècnic
- **Help Center**: Base de coneixement
- **Contact Support**: Suport tècnic directe
- **Community Forum**: Fòrum de la comunitat
- **Documentation**: Documentació tècnica completa

## 12. Bones Pràctiques

### Seguretat
- Utilitza contrasenyes fortes
- No comparteixis credencials
- Revisa regularment les connexions
- Mantén actualitzat el sistema

### Rendiment
- Comença amb models simples
- Optimitza hiperparàmetres gradualment
- Monitoritza l'ús de recursos
- Utilitza early stopping per evitar overfitting

### Col·laboració
- Documenta els teus experiments
- Comparteix resultats amb l'equip
- Utilitza noms descriptius per models
- Mantén organitzats els projectes

---

Aquesta guia cobreix totes les funcionalitats principals de MediNet. Per a informació més tècnica, consulta la documentació d'API o contacta amb el suport tècnic. 