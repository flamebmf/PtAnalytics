# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
import yaml

with open('D:\\www\\cam\\config\\cameras.yaml') as f:
    data = yaml.safe_load(f)

for cam in data['cameras']:
    print(f"ID: {cam['id']}")
    print(f"URL: {cam['rtsp_url']}")
    print()
