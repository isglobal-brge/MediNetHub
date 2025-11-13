// MediNet main JavaScript file

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Network visualization functions will be added here
    setupNetworkVisualization();
});

/**
 * Setup the network visualization if the container exists
 */
function setupNetworkVisualization() {
    const networkContainer = document.getElementById('network-visualization');
    if (!networkContainer) return;

    // Layer connection drawing will be implemented here
    drawLayerConnections();

    // Setup event listeners for layer node interactions
    setupLayerNodeInteractions();
}

/**
 * Draw connections between layer nodes
 */
function drawLayerConnections() {
    // This will be implemented when the network visualization is created
    console.log('Drawing layer connections');
}

/**
 * Setup event listeners for layer nodes
 */
function setupLayerNodeInteractions() {
    const layerNodes = document.querySelectorAll('.layer-node');
    
    layerNodes.forEach(node => {
        // Click to select a layer
        node.addEventListener('click', function() {
            // Remove active class from all nodes
            layerNodes.forEach(n => n.classList.remove('active'));
            
            // Add active class to clicked node
            this.classList.add('active');
            
            // Show parameters for this layer
            const layerId = this.dataset.layerId;
            showLayerParameters(layerId);
        });
    });
}

/**
 * Show parameters for a specific layer
 * @param {string} layerId The ID of the layer
 */
function showLayerParameters(layerId) {
    // This will be implemented when the layer parameters form is created
    console.log(`Showing parameters for layer ${layerId}`);
    
    // Hide all parameter forms
    const paramForms = document.querySelectorAll('.layer-parameters');
    paramForms.forEach(form => form.style.display = 'none');
    
    // Show the form for the selected layer
    const selectedForm = document.getElementById(`params-${layerId}`);
    if (selectedForm) {
        selectedForm.style.display = 'block';
    }
}

/**
 * Add a new layer to the network
 * @param {string} layerType The type of layer to add
 */
function addLayer(layerType) {
    // This will be implemented when the add layer functionality is created
    console.log(`Adding new layer of type ${layerType}`);
}

/**
 * Remove a layer from the network
 * @param {string} layerId The ID of the layer to remove
 */
function removeLayer(layerId) {
    // This will be implemented when the remove layer functionality is created
    console.log(`Removing layer ${layerId}`);
}

/**
 * Validate IP address
 * @param {string} ip IP address to validate
 * @returns {boolean} True if valid, false otherwise
 */
function validateIP(ip) {
    const ipPattern = /^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
    return ipPattern.test(ip);
}

/**
 * Validate port number
 * @param {string|number} port Port number to validate
 * @returns {boolean} True if valid, false otherwise
 */
function validatePort(port) {
    const portNum = parseInt(port);
    return !isNaN(portNum) && portNum >= 1 && portNum <= 65535;
}

/**
 * Test connection to server
 * @param {string} ip IP address
 * @param {number} port Port number
 */
function testConnection(ip, port) {
    if (!validateIP(ip)) {
        showMessage('Invalid IP address format', 'danger');
        return;
    }
    
    if (!validatePort(port)) {
        showMessage('Invalid port number', 'danger');
        return;
    }
    
    showMessage('Testing connection...', 'info');
    
    // This will be replaced with actual AJAX call
    setTimeout(function() {
        showMessage('Connection successful!', 'success');
    }, 1000);
}

/**
 * Show message to the user
 * @param {string} message Message to show
 * @param {string} type Message type (success, info, warning, danger)
 */
function showMessage(message, type = 'info') {
    const alertContainer = document.createElement('div');
    alertContainer.className = `alert alert-${type} alert-dismissible fade show`;
    alertContainer.role = 'alert';
    
    alertContainer.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const messagesContainer = document.querySelector('.messages');
    if (messagesContainer) {
        messagesContainer.appendChild(alertContainer);
        
        // Auto-hide after 5 seconds
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alertContainer);
            bsAlert.close();
        }, 5000);
    }
}

/**
 * Obtenir informació d'un dataset específic mitjançant la teva API real
 * @param {string|number} datasetId ID del dataset a obtenir
 * @returns {Promise} Promise amb les dades del dataset
 */
function getDatasetInfo(datasetId) {
    // Utilitzar l'endpoint correcte amb el paràmetre dataset_id
    return fetch(`/get_data_info?dataset_id=${datasetId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('No s\'ha pogut obtenir la informació del dataset');
            }
            return response.json();
        });
}

/**
 * Configurar el model basant-se en un dataset
 * @param {object} datasetInfo Informació del dataset
 */
function configureModelWithDataset(datasetInfo) {
    // Aquesta funció s'ha de cridar des del model_designer.html
    if (!window.modelLayers) {
        console.error("modelLayers no està definit. Assegureu-vos d'estar a la pàgina de disseny de models.");
        return;
    }
    
    // Les dades poden venir en una estructura diferent segons la teva API
    // Obtenir el nombre total de columnes i restar 1 per la columna target
    const totalColumns = parseInt(datasetInfo.num_columns) || 13;
    const inputFeatures = totalColumns - 1;
    
    console.log(`Configurant model amb ${inputFeatures} features d'entrada`);
    
    // Configurar capa d'entrada
    const inputLayer = window.modelLayers.find(l => l.id === 'input');
    if (inputLayer) {
        inputLayer.params.features = inputFeatures;
    }
    
    // Determinar característiques de sortida basant-se en el tipus de label
    let outputFeatures = 1; // Valor per defecte per a classificació binària
    
    // Actualitzar la capa de sortida
    const outputLayer = window.modelLayers.find(l => l.id === 'output');
    if (outputLayer) {
        outputLayer.params.features = outputFeatures;
    }
    
    // Actualitzar la visualització
    if (window.renderLayers) {
        window.renderLayers();
    }
    
    // Mostrar un missatge de confirmació
    showMessage(`Model configurat per dataset amb ${inputFeatures} features d'entrada`, 'success');
}

/**
 * Comprovar si hi ha un paràmetre de dataset a la URL i configurar el model
 * Aquesta funció s'ha de cridar des de model_designer.html
 */
function checkAndLoadDataset() {
    const urlParams = new URLSearchParams(window.location.search);
    const datasetId = urlParams.get('dataset');
    
    if (datasetId) {
        // Obtenir la informació del dataset de l'API real i configurar el model
        getDatasetInfo(datasetId)
            .then(datasetInfo => {
                configureModelWithDataset(datasetInfo);
                
                // Afegir badge que indiqui la connexió amb el dataset
                addDatasetBadge(datasetInfo);
            })
            .catch(error => {
                showMessage(`Error en carregar el dataset: ${error.message}`, 'danger');
            });
    }
}

/**
 * Afegir un badge indicant que el model està connectat a un dataset
 * @param {object} datasetInfo Informació del dataset
 */
function addDatasetBadge(datasetInfo) {
    const configForm = document.getElementById('model-config-form');
    if (!configForm) return;
    
    // Comprovar si ja existeix el badge
    if (!document.getElementById('dataset-badge')) {
        // Utilitzar el nom del dataset o un identificador alternatiu si el nom no està disponible
        const datasetName = datasetInfo.dataset_name || datasetInfo.name || `Dataset ${datasetInfo.id}`;
        
        const badgeHTML = `
            <div id="dataset-badge" class="alert alert-success mb-3">
                <i class="fas fa-database me-2"></i>Model connectat a dataset: <strong>${datasetName}</strong>
            </div>
        `;
        
        configForm.insertAdjacentHTML('afterbegin', badgeHTML);
    }
} 