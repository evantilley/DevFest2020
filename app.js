let express = require('express')
let upload = require("express-fileupload")
let mongoose = require('mongoose')
let bodyParser = require('body-parser')
let cookieParser = require('cookie-parser')
let csvtojson = require('csvtojson')


mongoose.connect('mongodb://localhost:27017/MachineLearning')
let db = mongoose.connection

var infoSchema = new mongoose.Schema({
    dataset: String,
    attributes: Array,
    predict: String
})

var Info = mongoose.model('Item', infoSchema, "Data")


db.once('open', function(){
    console.log("Connected to MongoDB")
})


let app = express()
app.use(upload())
app.use(bodyParser.json())
app.use(bodyParser.urlencoded({extended: true}))
app.use(cookieParser())

//bring in models

app.get('/', function(req, res){
    res.sendFile(__dirname + '/index.html')
})

app.get('/test', function(req, res){
    res.sendFile(__dirname + '/test.html')
})

app.post("/", function(req, res){


    console.log(req.body.info)




    if (req.files){
        var file = req.files.filename
        var myData = new Info(JSON.parse(req.body.info))
        console.log("sDFASDFSADFASDF")
        myData.dataset = file.name
        myData.save();
        console.log("file")
        var filename = file.name
        console.log("file name")
        console.log(filename)
        var file_name = ""
        file.mv("./" + filename, function(err){
            //res.render("./index.html")
            res.send("yeyee")
            const mongodb = require("mongodb").MongoClient;
            const csvtojson = require("csvtojson");
            
            // let url = "mongodb://username:password@localhost:27017/";
            let url = "mongodb://localhost:27017/";
            
            csvtojson()
              .fromFile("./" + filename)
              .then(csvData => {
            
                mongodb.connect(
                  url,
                  { useNewUrlParser: true, useUnifiedTopology: true },
                  (err, client) => {
                    if (err) throw err;
            
                    client
                      .db("MachineLearning")
                      .createCollection(filename)

                    client
                        .db("MachineLearning")
                        .collection(filename)
                        .insertMany(csvData, (err, res) => {
                            if (err) throw err;
                
                            console.log(`Inserted: ${res.insertedCount} rows`);
                            client.close();
                        });
                  }
                );
              });

              const fs = require('fs')

        const path = "./" + filename; 
        })     

    }
    var test = {dataset: "ye", attributes: file_name, predict: "yeye"}

})


app.listen(3000, function(){
    console.log("server started on port 3000")
})