let express = require('express')

let upload = require("express-fileupload")

let app = express()
app.use(upload())


app.get('/', function(req, res){
    res.sendFile(__dirname + '/index.html')
})

app.post("/", function(req, res){
    if (req.files){
        var file = req.files.filename
        var filename = file.name
        file.mv("C:/Users/etill/Documents/MachineLearning/" + filename, function(err){
            res.send("yeye")
        })
    }
})


app.listen(3000, function(){
    console.log("server started on port 3000")
})