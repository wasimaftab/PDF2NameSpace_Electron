// main.js

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

async function isGrobidServerRunning(url = "http://localhost:8070") {
  try {
    const response = await axios.get(`${url}/api/isalive`, { timeout: 2000 });
    return response.status === 200;
  } catch (error) {
    return false;
  }
}

ipcMain.handle('select-pdf-folder', async (event, namespace) => {
  // Check if GROBID server is running
  const grobidRunning = await isGrobidServerRunning();
  if (!grobidRunning) {
    // Send a message to the renderer process to inform the user
    event.sender.send('grobid-not-running');
    return; // Do not proceed further
  }

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
        // event.reply('processing-update', `Error: ${error}`);
        event.sender.send('processing-update', `Error: ${error}`);
      }
    });
}


function startPythonProcess(folderPath, namespace, event) {
  let pythonExecutablePath;
  let scriptPath;
  let args = [];

  if (isDev) {
    if (process.platform === 'win32') {
      pythonExecutablePath = 'C:\\Users\\yourusername\\Anaconda3\\envs\\yourenv\\python.exe';
    } else {
      pythonExecutablePath = '/home/wasim/anaconda3/envs/pdf2ns/bin/python';
    }

    scriptPath = path.join(__dirname, 'python', 'app.py');
    args = ['-u', scriptPath, folderPath, namespace];

  } else {
    if (process.platform === 'win32') {
      pythonExecutablePath = path.join(__dirname, 'py', 'app.exe');
    } else {
      pythonExecutablePath = path.join(__dirname, 'py', 'app');
    }

    args = [folderPath, namespace];
  }

  const env = Object.assign({}, process.env, {
    PINECONE_API_KEY: process.env.PINECONE_API_KEY,
    PINECONE_INDEX_NAME: process.env.PINECONE_INDEX_NAME,
  });

   // Spawn the Python process
   const pythonProcess = spawn(pythonExecutablePath, args, { env });

   pythonProcess.stdout.on('data', (data) => {
     const message = data.toString();
     event.sender.send('processing-update', message);
   });
 
   pythonProcess.stderr.on('data', (data) => {
     const error = data.toString();
     event.sender.send('processing-update', `Error: ${error}`);
   });

  // const pythonProcess = spawn(pythonExecutablePath, args, {
  //   env,
  //   stdio: ['ignore', 'pipe', 'pipe'],
  // });

  // pythonProcess.stdout.setEncoding('utf8');
  // pythonProcess.stdout.on('data', (data) => {
  //   data.split(/\r?\n/).forEach((line) => {
  //     if (line) {
  //       event.sender.send('processing-update', line);
  //     }
  //   });
  // });

  // pythonProcess.stderr.setEncoding('utf8');
  // pythonProcess.stderr.on('data', (data) => {
  //   data.split(/\r?\n/).forEach((line) => {
  //     if (line) {
  //       event.sender.send('processing-update', `Error: ${line}`);
  //     }
  //   });
  // });

  pythonProcess.on('close', (code) => {
    event.sender.send('processing-complete', `Python process exited with code ${code}`);
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
