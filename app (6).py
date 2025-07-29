from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from your app created with localtunnel!"

if __name__ == "__main__":
    app.run()
