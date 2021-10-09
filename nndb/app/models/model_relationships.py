# Import database and timestamp class
from app import db
from app.models.timestampmixin import TimestampMixin


# Neural Network Layer Data Model
class Layer(db.Model, TimestampMixin):
    """Neural Network Layer Data Model"""
    __tablename__ = 'layers'

    id = db.Column(db.Integer, primary_key=True)
    network_id = db.Column(db.Integer, db.ForeignKey('networks.id'))
    position = db.Column(db.Integer, nullable=False)
    activation = db.Column(db.String, nullable=False)
    

    owner = db.Column(db.String, nullable=False)
    owner_email = db.Column(db.String, nullable=False)

    def __init__(self, project_id, owner, email):
        self.project_id = project_id
        self.owner = owner
        self.owner_email = email

    def __repr__(self):
        return f"<Layer {self.id} on Network {self.network.id}>"

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "owner": self.owner,
            "owner_email": self.owner_email,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json


# Project Aliases Data Model
class Alias(db.Model, TimestampMixin):
    """Alias Data Model"""
    __tablename__ = 'aliases'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    alias = db.Column(db.String, nullable=False)

    def __init__(self, project_id, alias):
        self.project_id = project_id
        self.alias = alias

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "alias": self.alias,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Alias {self.alias} on Project ID {self.project.id}>"


# Project Indications Data Model
class Indication(db.Model, TimestampMixin):
    """Indication Data Model"""
    __tablename__ = 'indications'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    indication = db.Column(db.String, nullable=False)

    def __init__(self, project_id, indication):
        self.project_id = project_id
        self.indication = indication

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "indication": self.indication,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Indication {self.indication} on Project ID {self.project.id}>"


# Project Fill Finish Sites Data Model
class FFSite(db.Model, TimestampMixin):
    """F/F Site Data Model"""
    __tablename__ = 'ffsites'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    phase = db.Column(db.String)
    site = db.Column(db.String, nullable=False)
    line = db.Column(db.String)

    def __init__(self, project_id, site, line=None, phase=None):
        self.project_id = project_id
        self.site = site
        self.line = line
        self.phase = phase

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "site": self.site,
            "line": self.line,
            "phase": self.phase,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<FFSite {self.site} on Project ID {self.project.id}>"


# Project Accomplishments Data Model
class Accomplishment(db.Model, TimestampMixin):
    """Project Accomplishments Data Model"""
    __tablename__ = 'accomplishments'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    accomplishment = db.Column(db.Text, nullable=False)

    def __init__(self, project_id, accomplishment):
        self.project_id = project_id
        self.accomplishment = accomplishment

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "accomplishment": self.accomplishment,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Accomplishment {self.accomplishment} on  Project ID {self.project.id}>"


# Project Deliverables Data Model
class Deliverable(db.Model, TimestampMixin):
    """Project Deliverable Data Model"""
    __tablename__ = 'deliverables'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    deliverable = db.Column(db.Text, nullable=False)

    def __init__(self, project_id, deliverable):
        self.project_id = project_id
        self.deliverable = deliverable

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "deliverable": self.deliverable,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Deliverable {self.deliverable} on Project ID {self.project.id}>"


# Project Issues Data Model
class Issue(db.Model, TimestampMixin):
    """Project Issue Data Model"""
    __tablename__ = 'issues'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    issue = db.Column(db.Text, nullable=False)

    def __init__(self, project_id, issue):
        self.project_id = project_id
        self.issue = issue

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "issue": self.issue,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Issue {self.issue} on Project ID {self.project.id}>"


# Project Risks Data Model
class Risk(db.Model, TimestampMixin):
    """Project Risk Data Model"""
    __tablename__ = 'risks'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    risk = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.Text, nullable=False)

    def __init__(self, project_id, risk, level):
        self.project_id = project_id
        self.risk = risk
        self.risk_level = level

    def to_json(self, add_project_info=False):
        json = {
            "id": self.id,
            "project_id": self.project_id,
            "risk": self.risk,
            "risk_level": self.risk_level,
            "created": self.created,
            "updated": self.updated,
        }

        if add_project_info:
            prj_info = {
                "project_jnj_number": self.project.jnj_number,
                "project_top_alias": self.project.top_aliases(),
                "project_top_owner": self.project.top_owners(),
                "project_url": self.project.url_to_edit
            }

            json = {**json, **prj_info}

        return json

    def __repr__(self):
        return f"<Risk {self.risk} on Project ID {self.project.id}>"
