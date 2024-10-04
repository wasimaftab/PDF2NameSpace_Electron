const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');

// Enable live reload for all the files inside your project directory
require('electron-reload')(__dirname, {
  electron: require(`${__dirname}/node_modules/electron`)
});

let pythonProcess;

// Define isDev to determine the environment mode
const isDev = process.env.NODE_ENV === 'development';

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

// ipcMain.on('select-pdf-folder', async (event) => {
//   const result = await dialog.showOpenDialog({
//     properties: ['openDirectory'],
//   });
//   if (!result.canceled) {
//     const folderPath = result.filePaths[0];
//     event.reply('pdf-folder-selected', folderPath);

//     // Start the Python executable
//     startPythonProcess(folderPath, event);
//   }
// });

ipcMain.handle('select-pdf-folder', async (event, namespace) => {
  const result = await dialog.showOpenDialog({
    properties: ['openDirectory'],
  });
  if (!result.canceled) {
    const folderPath = result.filePaths[0];
    event.sender.send('pdf-folder-selected', folderPath);

    // Start the Python executable
    startPythonProcess(folderPath, namespace, event);
  }
});

function processPdfs(folderPath, namespace, event, retries = 5) {
  axios
    .post('http://127.0.0.1:8000/process-pdfs/', { folder_path: folderPath, namespace: namespace })
    .then((response) => {
      // event.reply('processing-complete', response.data.message);
      event.sender.send('processing-complete', response.data.message);
    })
    .catch((error) => {
      if (retries > 0) {
        // Wait and retry
        setTimeout(() => {
          processPdfs(folderPath, namespace, event, retries - 1);
        }, 1000);
      } else {
        console.error(error);
        event.reply('processing-update', `Error: ${error}`);
      }
    });
}


function startPythonProcess(folderPath, namespace, event) {
  if (isDev) {
    // Development mode: Assume FastAPI server is running separately
    processPdfs(folderPath, namespace, event);
  } else {
    // Production mode: Start the Python process as before
    let pythonExecutablePath;
    if (process.platform === 'win32') {
      pythonExecutablePath = path.join(__dirname, 'py', 'app.exe');
    } else {
      pythonExecutablePath = path.join(__dirname, 'py', 'app');
    }

    // Set environment variables
    const env = Object.assign({}, process.env, {
      PINECONE_API_KEY: process.env.PINECONE_API_KEY,
      PINECONE_INDEX_NAME: process.env.PINECONE_INDEX,
    });

    // Spawn the packaged executable
    pythonProcess = spawn(pythonExecutablePath, [folderPath], { env });

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
