project_root/
│
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry point
│   ├── models.py          # Database models (SQLAlchemy)
│   ├── schemas.py         # Pydantic schemas for request and response validation
│   ├── database.py        # Database connection and session management
│   ├── crud.py            # CRUD operations (create, read, update, delete)
│   ├── dependencies.py    # Dependencies for FastAPI routes
│   └── routes/            # Folder for FastAPI route modules
│       ├── patient.py     # Routes for handling patient data
│       └── results.py     # Routes for displaying results
│
├── ml_models/
│   ├── nlp_model/         # Folder containing NLP model files
│   │   ├── __init__.py
│   │   └── model.py       # NLP model loading and prediction logic
│   │
│   └── ml_model/          # Folder containing ML model files
│       ├── __init__.py
│       └── model.py       # ML model loading and prediction logic
│
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/             # HTML templates
│   ├── index.html         # Homepage template
│   ├── questionnaire.html # Questionnaire page template
│   └── results.html       # Results page template
│
├── tests/                 # Unit and integration tests
│   ├── __init__.py
│   └── test_routes.py
│
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
