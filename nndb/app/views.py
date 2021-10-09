# Imports
from app import db
from app.models.models import Collection, Neural_Network
from app.models.model_relationships import (
    Layer,
    Node
)
#from app.utilities.build_sorted_risks import sort_risks
from flask import Blueprint, g, redirect, render_template, request, url_for, jsonify

# Blueprint for the main application
bp = Blueprint('platform', __name__)

'''
MAIN APP PAGES
'''

# App route to an index page which contains links and instructions for how to use the application
@bp.route('/', methods=('GET',))
def index():

    # Get Database Objects
    #projects = Project.query.all()
    #projects_data = [project.to_json(True) for project in projects]

    # Render Page, Pass JSON Variables for Charting
    #return render_template('index.html', projects=projects_data)
    return render_template('index.html')

'''
# Simple endpoint for returning changelog data
@bp.route('/changelog_api', methods=('GET',))
def changelog_api():
    changelog = Changelog.query.all()
    changelog_data = []
    for change in reversed(changelog):
        project = Project.query.filter_by(id = change.project_id).first()
        changelog_data.append(change.to_json(project))

    return {"data": changelog_data}
'''

# Simple endpoint for project testing
@bp.route('/health_check', methods=('GET',))
def health_check():
    return "Healthy"
