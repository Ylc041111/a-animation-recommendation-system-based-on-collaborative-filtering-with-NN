const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const leftBtn = document.querySelector(".buttons .left")
const rightBtn = document.querySelector(".buttons .right")
const slide = document.querySelector("#slide")
const centerBtn = document.querySelector(".buttons .center")

let ErrorBox = document.querySelector("#ErrorBox")
let removeAll = document.querySelector('#removeAll');
let historyBack = document.querySelector('#historyBack');
let NameList = ['ScoreThreshold', 'Aired', 'PremieredSeason', 'Status', 'Source', 'Type', 'Duration', 'Episodes', 'Members', 'ScoreBy', 'Popularity', 'Favorites', 'Studios', 'Rating', 'Synopsis', 'Producers']
let ScoreThreshold = document.querySelector('#ScoreThreshold');
let Aired = document.querySelector('#Aired');
let PremieredSeason = document.querySelector('#PremieredSeason');
let Status = document.querySelector('#Status');
let Source = document.querySelector('#Source');
let Type = document.querySelector('#Type');
let Duration = document.querySelector('#Duration');
let Episodes = document.querySelector('#Episodes');
let Members = document.querySelector('#Members');
let ScoreBy = document.querySelector('#ScoredBy');
let Popularity = document.querySelector('#Popularity');
let Favorites = document.querySelector('#Favorites');
let Studios = document.querySelector('#Studios');
let Rating = document.querySelector('#Rating');
let Synopsis = document.querySelector('#Synopsis');
let Producers = document.querySelector('#Producers');
let all = document.querySelector('.all');

let openClick = true
let index = 1
let totalArgs = 16

// input输入enter时触发rightBtn
document.querySelectorAll('input').forEach((item, index, _array) => {
    item.addEventListener('keydown', function (e) {
        if (openClick && e.key === "Enter") {
            rightBtn.click();
            // 聚焦下一个item input
            if (index == _array.length - 1) index = 0;
            _array[index + 1].focus({ preventScroll: true });
        }
    });
})

rightBtn.addEventListener("click", () => {
    if (openClick) {
        openClick = false
        if (index == totalArgs) index = 0;
        else index++;
        if (index == 0) {
            centerBtn.innerText = `submit`
            let ScoreThresholdValue = ScoreThreshold.value;
            let AiredValue = Aired.value;
            let PremieredSeasonValue = PremieredSeason.value;
            let StatusValue = Status.value;
            let SourceValue = Source.value;
            let TypeValue = Type.value;
            let DurationValue = Duration.value;
            let EpisodesValue = Episodes.value;
            let MembersValue = Members.value;
            let ScoreByValue = ScoreBy.value;
            let PopularityValue = Popularity.value;
            let FavoritesValue = Favorites.value;
            let StudiosValue = Studios.value;
            let RatingValue = Rating.value;
            let SynopsisValue = Synopsis.value;
            let ProducersValue = Producers.value;

            let allargs = []

            allargs.push(ScoreThresholdValue)
            allargs.push(AiredValue)
            allargs.push(PremieredSeasonValue)
            allargs.push(StatusValue)
            allargs.push(SourceValue)
            allargs.push(TypeValue)
            allargs.push(DurationValue)
            allargs.push(EpisodesValue)
            allargs.push(MembersValue)
            allargs.push(ScoreByValue)
            allargs.push(PopularityValue)
            allargs.push(FavoritesValue)
            allargs.push(StudiosValue)
            allargs.push(RatingValue)
            allargs.push(SynopsisValue)
            allargs.push(ProducersValue)

            let argsHTML = ""
            allargs.forEach((item, index) => {
                argsHTML += `<p>${index + 1}. ${NameList[index]}: ${item}</p>`
            })
            all.innerHTML = argsHTML
        }
        else centerBtn.innerText = `${index} / ${totalArgs}`;
        const items = document.querySelectorAll(".item")
        slide.appendChild(items[0])
        setTimeout(() => openClick = true, 1000)
    }
})

leftBtn.addEventListener("click", () => {
    if (openClick) {
        openClick = false
        if (index == 0) index = totalArgs;
        else index--;
        if (index == 0) {
            centerBtn.innerText = `submit`
            let ScoreThresholdValue = ScoreThreshold.value;
            let AiredValue = Aired.value;
            let PremieredSeasonValue = PremieredSeason.value;
            let StatusValue = Status.value;
            let SourceValue = Source.value;
            let TypeValue = Type.value;
            let DurationValue = Duration.value;
            let EpisodesValue = Episodes.value;
            let MembersValue = Members.value;
            let ScoreByValue = ScoreBy.value;
            let PopularityValue = Popularity.value;
            let FavoritesValue = Favorites.value;
            let StudiosValue = Studios.value;
            let RatingValue = Rating.value;
            let SynopsisValue = Synopsis.value;
            let ProducersValue = Producers.value;

            let allargs = []

            allargs.push(ScoreThresholdValue)
            allargs.push(AiredValue)
            allargs.push(PremieredSeasonValue)
            allargs.push(StatusValue)
            allargs.push(SourceValue)
            allargs.push(TypeValue)
            allargs.push(DurationValue)
            allargs.push(EpisodesValue)
            allargs.push(MembersValue)
            allargs.push(ScoreByValue)
            allargs.push(PopularityValue)
            allargs.push(FavoritesValue)
            allargs.push(StudiosValue)
            allargs.push(RatingValue)
            allargs.push(SynopsisValue)
            allargs.push(ProducersValue)

            let argsHTML = ""
            allargs.forEach((item, index) => {
                argsHTML += `<p>${index + 1}. ${NameList[index]}: ${item}</p>`
            })
            all.innerHTML = argsHTML
        }
        else centerBtn.innerText = `${index} / ${totalArgs}`;
        const items = document.querySelectorAll(".item")
        slide.prepend(items[items.length - 1])
        openClick = true
    }
})

let submit = document.querySelector('#submit');
let resultBox = document.querySelector('.resultBox');
let detail = document.querySelector('#detail');
let load = document.querySelector('.load');
let loadText = document.querySelector('.loadText');
let back = document.querySelector('.back');

submit.onclick = function () {
    if (index != 0) return;
    let beginTime = new Date().getTime();
    let WaitTime = document.querySelector('.WaitTime');

    let timer = setInterval(() => {
        let endTime = new Date().getTime();
        // 以 xx : xx 格式显示
        let time = Math.floor((endTime - beginTime) / 1000);
        let minutes = Math.floor(time / 60);
        if (minutes < 10) minutes = "0" + minutes;
        let seconds = time % 60;
        if (seconds < 10) seconds = "0" + seconds;
        WaitTime.innerText = `${minutes} : ${seconds}`;
    }, 1000)
    // 将resultBox的display属性设置为flex
    resultBox.style.display = "flex";
    historyBack.style.display = "none";
    WaitTime.style.display = "block";
    removeAll.style.display = "none";
    // 依次拿到index.html中input的值
    let ScoreThresholdValue = ScoreThreshold.value;
    let AiredValue = Aired.value;
    let PremieredSeasonValue = PremieredSeason.value;
    let StatusValue = Status.value;
    let SourceValue = Source.value;
    let TypeValue = Type.value;
    let DurationValue = Duration.value;
    let EpisodesValue = Episodes.value;
    let MembersValue = Members.value;
    let ScoreByValue = ScoreBy.value;
    let PopularityValue = Popularity.value;
    let FavoritesValue = Favorites.value;
    let StudiosValue = Studios.value;
    let RatingValue = Rating.value;
    let SynopsisValue = Synopsis.value;
    let ProducersValue = Producers.value;

    let data = [{
        ScoreThreshold: ScoreThresholdValue,
        Aired: AiredValue,
        PremieredSeason: PremieredSeasonValue,
        PremieredYear: AiredValue,
        Status: StatusValue,
        Source: SourceValue,
        Type: TypeValue,
        Duration: DurationValue,
        Episodes: EpisodesValue,
        Members: MembersValue,
        ScoreBy: ScoreByValue,
        Popularity: PopularityValue,
        Favorites: FavoritesValue,
        Studios: StudiosValue,
        Rating: RatingValue,
        Synopsis: SynopsisValue,
        Producers: ProducersValue
    }]

    // 将data写入userData.json
    // online
    const wroad = path.join(__dirname, '../userData.json')
    // dev
    // const wroad = path.join(__dirname, 'userData.json')
    // console.log(wroad)

    fs.writeFile(wroad, JSON.stringify(data), function (err) {
        if (err) throw err;
        console.log('Data successful written to file');
    })

    // 触发python代码运行
    // online
    const pyroad = path.join(__dirname, '../main.py');
    // dev
    // const pyroad = path.join(__dirname, 'main.py');
    // console.log(pyroad)

    const pythonProcess = spawn('python', [pyroad]);

    pythonProcess.stdout.on('data', (data) => {
        // console.log(`Received data from Python script: ${data}`);
        // 读取resultData.json文件中的数据
        // online
        const rroad = path.join(__dirname, '../resultData.json')
        // dev
        // const rroad = path.join(__dirname, 'resultData.json')
        // console.log(rroad)
        let resultData = []
        let historyData = []
        fs.readFile(rroad, 'utf8', (err, data) => {
            if (err) {
                clearInterval(timer);
                resultBox.style.display = "none";
                ErrorBox.style.display = "flex";
                setTimeout(() => {
                    ErrorBox.style.display = "none";
                    historyBack.style.display = "block";
                    removeAll.style.display = "block";
                }, 2000)
            }
            // 将数据转换为JSON对象 
            resultData = JSON.parse(data);
            // console.log(resultData)

            let Html = ""
            resultData.forEach(item => {
                historyData.push(item)
                // console.log(historyData)
                Html += `<div><p>${item.Name}</p>
                    <p>${item.OtherName}</p>
                    <p>${item.Genres}</p>
                    <p>${item.Score}</p>
                    <img src="${item.ImageURL}"></img></div><br><br><br>
                    `
            })

            // console.log(Html)
            setTimeout(() => {
                const hroad = path.join(__dirname, '../historyData.json')
                // console.log(hroad)
                // const hroad = path.join(__dirname, 'historyData.json')
                // 读取historyData.json文件中的数据
                fs.readFile(hroad, 'utf8', (err, data) => {
                    if (err) {
                        clearInterval(timer);
                        fs.writeFile(hroad, JSON.stringify(historyData), function (err) {
                            if (err) throw err;
                            console.log('Data successful written to file');
                        })
                        return;
                    }
                    if (data == "") {
                        // 将数据写入historyData.json文件
                        clearInterval(timer);
                        fs.writeFile(hroad, JSON.stringify(historyData), function (err) {
                            if (err) throw err;
                            console.log('Data successful written to file');
                        })
                        return;
                    }
                    let temp = JSON.parse(data)

                    temp.forEach(item => {
                        historyData.push(item)
                    });

                    // 将数据写入historyData.json文件
                    fs.writeFile(hroad, JSON.stringify(historyData), function (err) {
                        if (err) throw err;
                        console.log('Data successful written to file');
                    })
                })

                clearInterval(timer);
                WaitTime.innerText = `00 : 00`;
                WaitTime.style.display = "none";
                detail.style.display = "block";
                back.style.display = "block";
                load.style.display = "none";
                loadText.style.display = "none";
                detail.innerHTML = Html
                detail.scrollTo(0, -1);
            }, 2000)
        })
    })

    pythonProcess.stderr.on('data', (data) => {
        clearInterval(timer);
        resultBox.style.display = "none";
        ErrorBox.style.display = "flex";
        setTimeout(() => {
            ErrorBox.style.display = "none";
            historyBack.style.display = "block";
            removeAll.style.display = "block";
        }, 2000)
        console.error(`Error from Python script: ${data}`);
    })

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            resultBox.style.display = "none";
            ErrorBox.style.display = "flex";
            setTimeout(() => {
                ErrorBox.style.display = "none";
                historyBack.style.display = "block";
                removeAll.style.display = "block";
            }, 2000)
        }
        console.log(`Python script exited with code ${code}`);
    })

    back.onclick = function () {
        // 将resultBox的display属性设置为none
        resultBox.style.display = "none";
        detail.style.display = "none";
        back.style.display = "none";
        load.style.display = "block";
        loadText.style.display = "block";
        historyBack.style.display = "block";
        removeAll.style.display = "block";
    }
}

let needknow = document.querySelector('.needknow');
let confirm = document.querySelector('.confirm');
confirm.onclick = function () {
    needknow.style.display = "none";
}



historyBack.onclick = function () {
    // 将resultBox的display属性设置为flex
    resultBox.style.display = "flex";

    const hroad = path.join(__dirname, '../historyData.json')
    // 读取historyData.json文件中的数据
    fs.readFile(hroad, 'utf8', (err, data) => {
        if (err) {
            historyBack.style.display = "none";
            removeAll.style.display = "none";
            detail.innerText = "暂无历史记录"

            setTimeout(() => {
                detail.style.display = "block";
                back.style.display = "block";
                load.style.display = "none";
                loadText.style.display = "none";
                detail.scrollTo(0, -1);
            }, 1000)

            back.onclick = function () {
                // 将resultBox的display属性设置为none
                resultBox.style.display = "none";
                detail.style.display = "none";
                back.style.display = "none";
                load.style.display = "block";
                loadText.style.display = "block";
                historyBack.style.display = "block";
                removeAll.style.display = "block";
            }
            return;
        }
        if (data == "") {
            historyBack.style.display = "none";
            removeAll.style.display = "none";
            detail.innerText = "暂无历史记录"

            setTimeout(() => {
                detail.style.display = "block";
                back.style.display = "block";
                load.style.display = "none";
                loadText.style.display = "none";
                detail.scrollTo(0, -1);
            }, 1000)
        } else {
            historyBack.style.display = "none";
            removeAll.style.display = "none";
            let temp = JSON.parse(data)
            let Html = ""
            temp.forEach(item => {
                Html += `<div><p>${item.Name}</p>
                    <p>${item.OtherName}</p>
                    <p>${item.Genres}</p>
                    <p>${item.Score}</p>
                    <img src="${item.ImageURL}"></img></div><br><br><br>
                    `
            })

            setTimeout(() => {
                detail.style.display = "block";
                back.style.display = "block";
                load.style.display = "none";
                loadText.style.display = "none";
                detail.innerHTML = Html
                detail.scrollTo(0, -1);
            }, 3000)
        }

        back.onclick = function () {
            // 将resultBox的display属性设置为none
            resultBox.style.display = "none";
            detail.style.display = "none";
            back.style.display = "none";
            load.style.display = "block";
            loadText.style.display = "block";
            historyBack.style.display = "block";
            removeAll.style.display = "block";
        }

    })
}

removeAll.onclick = function () {
    // 清空input中的内容
    ScoreThreshold.value = '';
    Aired.value = '';
    PremieredSeason.value = '';
    Status.value = '';
    Source.value = '';
    Type.value = '';
    Duration.value = '';
    Episodes.value = '';
    Members.value = '';
    ScoreBy.value = '';
    Popularity.value = '';
    Favorites.value = '';
    Studios.value = '';
    Rating.value = '';
    Synopsis.value = '';
    Producers.value = '';

    let ScoreThresholdValue = ScoreThreshold.value;
    let AiredValue = Aired.value;
    let PremieredSeasonValue = PremieredSeason.value;
    let StatusValue = Status.value;
    let SourceValue = Source.value;
    let TypeValue = Type.value;
    let DurationValue = Duration.value;
    let EpisodesValue = Episodes.value;
    let MembersValue = Members.value;
    let ScoreByValue = ScoreBy.value;
    let PopularityValue = Popularity.value;
    let FavoritesValue = Favorites.value;
    let StudiosValue = Studios.value;
    let RatingValue = Rating.value;
    let SynopsisValue = Synopsis.value;
    let ProducersValue = Producers.value;

    let allargs = []

    allargs.push(ScoreThresholdValue)
    allargs.push(AiredValue)
    allargs.push(PremieredSeasonValue)
    allargs.push(StatusValue)
    allargs.push(SourceValue)
    allargs.push(TypeValue)
    allargs.push(DurationValue)
    allargs.push(EpisodesValue)
    allargs.push(MembersValue)
    allargs.push(ScoreByValue)
    allargs.push(PopularityValue)
    allargs.push(FavoritesValue)
    allargs.push(StudiosValue)
    allargs.push(RatingValue)
    allargs.push(SynopsisValue)
    allargs.push(ProducersValue)

    let argsHTML = ""
    allargs.forEach((item, index) => {
        argsHTML += `<p>${index + 1}. ${NameList[index]}: ${item}</p>`
    })
    all.innerHTML = argsHTML

}