const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let pythonProcess;

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
    },
  });

  win.loadFile('index.html');
}

ipcMain.on('select-pdf-folder', async (event) => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  if (!result.canceled) {
    const folderPath = result.filePaths[0];
    event.reply('pdf-folder-selected', folderPath);

    // Start the Python executable
    startPythonProcess(folderPath, event);
  }
});

function startPythonProcess(folderPath, event) {
  // Determine the path to the Python executable based on the platform
  let pythonExecutablePath;
  if (process.platform === 'win32') {
    pythonExecutablePath = path.join(__dirname, 'py', 'app.exe');
  } else if (process.platform === 'darwin') {
    pythonExecutablePath = path.join(__dirname, 'py', 'app');
  } else if (process.platform === 'linux') {
    pythonExecutablePath = path.join(__dirname, 'py', 'app');
  }

  // Spawn the Python process
  pythonProcess = spawn(pythonExecutablePath, [folderPath]);

  pythonProcess.stdout.on('data', (data) => {
    const message = data.toString();
    event.reply('processing-update', message);
  });

  pythonProcess.stderr.on('data', (data) => {
    const error = data.toString();
    event.reply('processing-update', `Error: ${error}`);
  });

  pythonProcess.on('close', (code) => {
    event.reply('processing-complete', `Python process exited with code ${code}`);
    pythonProcess = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
