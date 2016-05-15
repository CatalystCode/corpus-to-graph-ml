from flask import Flask
from flask import request
from flask import jsonify
from flask import Response
from sklearn.externals import joblib
import json
import scorer

# convert the scorer result to the response format expected by the client
# TODO: move to another module
def scorer_result_to_response_format(scoring_results_and_entities):
    response_result = { 'modelVersion' : scorer.get_version(), 'relations' : []}
    
    for tup in scoring_results_and_entities:
        relation = {}
        scorer_result = tup[0]
        relation['entities'] = tup[1]
        max_score = -1
        max_score_class = ""
        scores_list = []
        for i in range(len(scorer_result)):
            if (scorer_result[i] > max_score):
                max_score = scorer_result[i]
                max_score_class = str(i)
                
        relation['classification'] = max_score_class
        relation['score'] = max_score
        response_result['relations'].append(relation)
    
    return response_result

app = Flask(__name__)

@app.route('/')
def api_root():
    return 'Relation classification service'

@app.route('/score', methods = ['POST'])
def score():
    if request.headers['Content-Type'] != 'application/json':
        resp = Response('Unssuported content type, expected application/json', status=500);
        return resp
    if (not request.json.has_key('text')):
        resp = Response('Bad request: missing "text" field in JSON body', status=500);
        return resp
    if (not request.json.has_key('entities')):
        resp = Response('Bad request: missing "entities" field in JSON body', status=500);
        return resp
    
    text = request.json['text']
    entities = request.json['entities']
    try:
        scorerResult = scorer.evaluate_score(text, entities)
        resp = jsonify(scorer_result_to_response_format(scorerResult))
        resp.status_code = 200
        return resp
    except Exception as e:
        resp = Response("Internal Server Error: %s"%e, status = 500)
        return resp
    
@app.route('/updatemodel', methods = ['POST'])
def update_model():
    if request.headers['Content-Type'] != 'application/json':
        resp = Response('Unssuported content type, expected application/json', status=500);
        return resp
    if (not request.json.has_key('path')):
        resp = Response('Bad request: missing "path" field in JSON body', status=500);
        return resp
    
    path = request.json['path']
    try:
        scorer.load_model_from_url(path)
        resp = Response("", status=200);
        return resp
    except Exception as e:
        resp = Response("Internal Server Error: %s"%e, status = 500)
        return resp
    
    
if __name__ == '__main__':
    app.run()