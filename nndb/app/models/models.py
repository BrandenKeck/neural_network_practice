import time
from app import db
from app.models.timestampmixin import TimestampMixin
from app.models.project_relationships import (
    Layer,
    Node
)

class Collection(db.Model, TimestampMixin):
    """Data Model for a ML Model Collection"""
    __tablename__ = 'collections'

    # Single value attributes
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String, nullable=False)
    name = db.Column(db.String, nullable=False)

    # Many-to-one relationships
    neural_networks = db.relationship('Neural_Network', backref='collection', lazy=True)

    # Initialize a colloction
    def __init__(self, name):
        self.name = name
        self.uuid = "xyz"
        db.session.add(self)
        db.session.commit()

    # Add Neural Network to Collection
    def add_neural_network(self, nn):
        self.neural_networks.append(nn)

class Neural_Network(db.Model, TimestampMixin):
    """Project Data Model"""
    __tablename__ = 'neural_networks'

    # Required Attributes
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String, nullable = False)
    name = db.Column(db.String, nullable=False)
    learning_rate = db.Column(db.Float, nullable=False)
    optimizer = db.Column(db.String, nullable=False)

    # Customizations
    huber_delta = db.Column(db.Float)

    # Network Layers
    layers = db.relationship('Layer', backref='neural_network', lazy=True)

    # Initialize a neural network
    def __init__(self, name, lr, opt):
        self.uuid = "xyz"
        self.name = name
        self.learning_rate = lr
        self.optimizer = opt
        db.session.add(self)
        db.session.commit()

    # Update Network
    def update():
        for layer in self.layers:
            layer.update()

    # Save Network
    def update_settings(lr=None, opt=None):
        if lr is not None: self.learning_rate = lr
        if opt is not None: self.optimizer = opt

    # Add Layer to the Network
    def add_layer(self, ):
        self.layers.append(Layer(self.id, ))

    # Update Layer within Network Instance
    def update_layer(self, id, ):

        # Update Layer
        layer = Layer.query.filter_by(id = id).first()
        layer.update()
        db.session.add(layer)
        db.session.commit()

    # Convert datamodel to JSON for API endpoints
    def to_json(self, relations=False):
        json = {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
            'learning_rate': self.learning_rate,
            'optimizer': self.optimizer,
            'huber_delta': self.huber_delta,
            'created': self.created,
            'updated': self.updated,
            'layers': [layer.to_json() for layer in self.layers]
        }

        return json

    # Print to terminal formatting
    def __repr__(self):
        return f"<Neural Network {self.name}: {self.uuid}>"
