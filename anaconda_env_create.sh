mamba create --name dashboard_pipeline10 pandas numpy plotly dash sqlalchemy psycopg2 pytest hypothesis

conda activate dashboard_pipeline10

mamba install -c conda-forge dash-bootstrap-components

pip install pyarrow
pip install pandera

pip install sqlalchemy_utils

pip install dash_daq

conda env export > environment.yml
