# ⚡ n8n — Workflow-Engine (Hostinger)

[![n8n](https://img.shields.io/badge/n8n-Workflow%20Automation-ea4b71?logo=n8n)](https://n8n.io)
[![Docker](https://img.shields.io/badge/Docker-Hostinger-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-Sustainable%20Use-green)](LICENSE)

**Selbstgehostete n8n-Instanz auf Hostinger VPS** mit benutzerdefinierten Workflows für KI-gestützte Automatisierung.

> n8n ist eine fair-code Workflow-Automation-Plattform mit 400+ Integrationen. Diese Instanz läuft auf einem Hostinger VPS und steuert zentrale Automatisierungen für das KI-Ökosystem.

---

## ✨ Features

- **📧 Email Agent:** IMAP-Trigger → Spam-Filter → KI-Klassifikation → Labeling
- **🩺 Vitalkontrolle:** Health-Checks für alle Dienste mit Alerting
- **🔗 400+ Integrationen:** REST, Webhook, Datenbanken, KI-Modelle
- **🐳 Docker-Deployment:** Läuft auf Hostinger VPS mit Cloudflare Tunnel
- **📊 Workflow-Versionierung:** Alle Workflows als JSON in Git

---

## 🚀 Deployment

Diese Instanz läuft auf einem **Hostinger VPS** via Docker:

```bash
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  -e N8N_SECURE_COOKIE=false \
  n8nio/n8n
```

Erreichbar unter: [n8n-newq.srv1741927.hstgr.cloud](https://n8n-newq.srv1741927.hstgr.cloud)

---

## 📋 Workflows

| Workflow | Datei | Beschreibung |
|---|---|---|
| 📧 **Email Agent** | `workflows/Email Agent.json` | IMAP-Trigger, Spam-Filter, KI-Klassifikation |
| 🩺 **Vitalkontrolle** | `workflows/Vitalkontrolle.json` | Health-Checks für alle Dienste |

---

## 🧱 Tech-Stack

| Komponente | Technologie |
|---|---|
| **Plattform** | n8n (fair-code) |
| **Hosting** | Hostinger VPS |
| **Tunnel** | Cloudflare Tunnel |
| **Container** | Docker |
| **Sprache** | TypeScript (n8n Core) |

---

## 📁 Projektstruktur

```
n8n/
├── workflows/
│   ├── Email Agent.json          # Email-Klassifikation & Routing
│   └── Vitalkontrolle.json       # Service-Health-Monitoring
└── packages/                     # n8n Core (upstream)
```

---

## 👤 Autor

**Mark Baumann** — [GitHub](https://github.com/mark-baumann) · [markb.de](https://markb.de)

---

*n8n ist ein Produkt von [n8n.io](https://n8n.io) unter Sustainable-Use-Lizenz.*
