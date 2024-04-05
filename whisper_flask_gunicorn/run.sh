#!/bin/bash
cd src/
gunicorn -c  gunicorn.conf.py app:app

