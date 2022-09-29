FROM python:slim
WORKDIR /app
# pipenv
RUN pip install pipenv
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY . .

CMD ["streamlit", "run", "Dashboard.py"]