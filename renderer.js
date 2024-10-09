// renderer.js
document.getElementById('select-folder').addEventListener('click', () => {
  // Get the namespace value from the textarea
  const namespaceInput = document.getElementById('namespace');
  const namespace = namespaceInput.value.trim();

  if (!namespace) {
    Swal.fire({
      icon: 'error',
      title: 'Empty Namespace',
      text: 'Namespace must not be empty string',
  });
    return;
  } else {
    console.log("namespace = " + namespace);
  }   
    window.electronAPI.selectPdfFolder(namespace);
  });

  window.electronAPI.onGrobidNotRunning(() => {
    Swal.fire({
      icon: 'error',
      title: 'GROBID Server Not Running',
      text: 'Please start the GROBID server before proceeding.',
      confirmButtonText: 'OK',
    });
  });
  
  window.electronAPI.onPdfFolderSelected((event, folderPath) => {
    const output = document.getElementById('output');
    output.textContent += `Selected folder: ${folderPath}\nProcessing...`;
  });
  
  window.electronAPI.onProcessingUpdate((event, message) => {
    const output = document.getElementById('output');
    output.textContent += `\n${message}`;
    output.scrollTop = output.scrollHeight; // Auto-scroll
  });
  
  window.electronAPI.onProcessingComplete((event, message) => {
    const output = document.getElementById('output');
    output.textContent += `\n${message}\nProcessing complete.\n############\n############\n`;
  });
  