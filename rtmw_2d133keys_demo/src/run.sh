#!/bin/bash
gunicorn -c gunicorn.conf.py app:app
