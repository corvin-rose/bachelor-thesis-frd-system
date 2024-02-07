## Backend

### Tests ausführen

Einfach
```
python manage.py test api
```
Detailliert
```
python manage.py test api --verbosity 2
```
Testabdeckung
```
coverage run --source=core,port manage.py test api
```
```
coverage report
```