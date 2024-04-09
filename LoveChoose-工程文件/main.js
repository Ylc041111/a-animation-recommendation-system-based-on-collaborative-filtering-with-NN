const { app, BrowserWindow, Menu } = require('electron')
const path = require('node:path')

// 屏蔽掉了devtools
Menu.setApplicationMenu(null)

function createWindow() {
    const win = new BrowserWindow({
        width: 1500,
        height: 800,
        icon: path.join(__dirname, 'logo/WLOP_squre.jpg'),
        webPreferences: {
            nodeIntegration: true,
            enableRemoteModule: true,
            contextIsolation: false,
            devTools: true
        }
    })
    win.loadFile('index.html')
    // win.webContents.openDevTools()
}

app.whenReady().then(() => {
    createWindow()

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow()
        }
    })
})

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit()
    }
})