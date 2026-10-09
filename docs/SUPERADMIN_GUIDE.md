# Superadmin guide

The platform team runs every organization from the **Platform Console** at `/platform/`, reached from the **Platform Console** link in the sidebar (only shown to people with console access). It has its own menu, a status bar (environment, maintenance, AI provider health) and its own Ctrl+K search across organizations, users and API keys; **Exit to workspace** goes back. Every console action is written to the **audit log**.

To make the first superadmin: `python manage.py createsuperuser` (or `docker compose exec web python manage.py createsuperuser`). In Docker you can also set `DJANGO_SUPERUSER_USERNAME` and `DJANGO_SUPERUSER_PASSWORD` in `.env.docker` and it's created on start.

## Console roles

| | Owner (superadmin) | Support | Finance |
| --- | :-: | :-: | :-: |
| Dashboard, organizations (view) | ✓ | ✓ | ✓ |
| Change organizations, plans, users, settings, AI providers | ✓ | | |
| Support mode (read-only "view as this organization") | ✓ | ✓ | |
| Users, security centre, audit log, system health | ✓ | view | |
| Plans and usage CSV | ✓ | | view |

Give someone a role on **Users** (the *Console* column). Making someone a superadmin, or removing it, asks you to type their username.

## Console pages

| Page | What you can do |
| --- | --- |
| **Dashboard** | Questions, active organizations and people, failure and backup rates for 7/30/90 days against the previous period; who wrote the answers (each AI provider's share, speed, failures, last error); every organization's usage; usage CSV |
| **Organizations** | Every organization with its plan and usage. Open one to: apply a **plan**; set limits and model; manage members; revoke keys (one or all); re-index failed documents; **View as this organization** (support mode); **Export data**; **suspend** (with a reason) or **schedule deletion** |
| **Plans** | Packages of limits: questions per month, documents, members, whether cloud AI is allowed |
| **Users** | Every account: organizations, console role, two-factor status. Reset a password (one-time temporary password), deactivate, grant or remove superadmin, reset two-factor |
| **AI providers** | Which AI models write answers and in what order, with each one's answers and failures. See [AI_PROVIDERS.md](AI_PROVIDERS.md) |
| **Security centre** | Console access and two-factor status, the two-factor requirement, failed sign-ins and pauses, every API key flagged when never used, unused 90+ days or older than 180 days |
| **Audit log** | Filter by action, person, organization, text and dates; before/after for changes; CSV export |
| **Platform settings** | Platform-wide settings, below |
| **System health** | Live status of the database, vector store, embedding model and AI providers, with a **Test** button for each |
| **Console guide** | A short in-app version of this page |

## Organization lifecycle

- **Plans** set the monthly question limit at once; document and member caps are checked when admins upload or add people, and *cloud AI not allowed* limits answers to local providers and quoting.
- **Suspend** needs a reason: members can't open the organization and its API keys stop. **Activate** undoes it.
- **Schedule deletion** (type the name) takes it offline now and deletes everything after **30 days**; **Cancel deletion** restores it. Deletions are carried out when the dashboard is opened, or by the `purge_organizations` command (run it daily from cron in production). **Delete now** skips the wait.
- **Export data** downloads a zip: the organization's members, documents list, API keys (no secrets), topics and every conversation as JSON, plus the original files.

## Support mode

**View as this organization** opens its workspace as an admin would see it, read-only: every change (uploads, settings, asking questions) is refused, a banner says so, and starting and leaving are recorded in the audit log. **Exit support mode** returns to the organization's console page.

## Two-factor sign-in

Set it up under your name → **Two-factor sign-in**: scan the QR code with an authenticator app (Google Authenticator, Microsoft Authenticator, 1Password…), enter the code, and save the 10 backup codes. From then on, signing in asks for a code. In the **Security centre**, owners can **require** two-factor for everyone with console access (set up your own first). Someone who lost their phone and codes: **Users → Reset 2FA**, or `python manage.py reset_2fa <username>`.

## Platform settings

`/platform/settings/` holds settings that used to need an `.env` edit and a restart. Changes apply right away (other server processes pick them up within 30 seconds). Each change is audit-logged as "label: old → new". Each tab has **Restore defaults**, which puts it back to the `.env` values.

| Tab | Settings |
| --- | --- |
| Access & security | Allow self sign-up; how long *Remember me* keeps people signed in; failed sign-ins before a pause; pause length |
| New organizations | Default Gemini model; monthly question limit; passages per answer; conversation memory; knowledge-gap threshold. These are starting values: existing organizations keep their own. |
| Documents & answers | Largest upload; quote the documents when every AI provider fails; how long to reuse answers to repeated questions (0 turns reuse off) |
| Announcement | A banner at the top of every page for signed-in people, as information or a warning. People can dismiss it for their session. |
| Maintenance | Maintenance mode and its message |

### Maintenance mode

When it's on, everyone except superadmins sees a "We'll be back soon" page with your message, and the API returns `503` with the message. Superadmins can still sign in, use the portal and switch it off, and they see a reminder banner. Turning it on asks for confirmation.

### Sign-in protection

After the set number of wrong passwords for one username from one network address, sign-in for that pair pauses for the set time. Counting per address means a stranger can't lock a real person out from elsewhere. The moment a pause starts, it's recorded in the audit log as `security.lockout`.

## Routine checks

- **Weekly:** Overview's attention list; System health (run the AI test); Audit log for unexpected `security.lockout` entries.
- **When answers fail:** check AI providers. "Using backup" in an organization's sidebar means its first provider failed and a later one (or quoting) answered.
- **Before planned work:** set an Announcement a day ahead, then switch on Maintenance mode during the work.

## Passwords

Reset a user's password from **Users** (shows a one-time temporary password), or on the server with `python manage.py changepassword <username>`. Never send passwords through shared documents; give them to the person directly and ask them to change it after signing in.
