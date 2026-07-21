# Meno — приватность и 152-ФЗ: дизайн-решения и целевая архитектура

**Дата:** 2026-07-21 · **Версия дизайна:** 1.0
**Репозитории:** backend `RAG-Core` (текущий), frontend `Meno-Web` (`/Users/sckwoky/PycharmProjects/Meno-Web`)
**Источник требований:** ТЗ «персональные данные, согласия и история диалогов Meno». Этот документ фиксирует **решения и дельты поверх ТЗ**; полные требования — в ТЗ.

Согласование НГУ пройдено: НГУ — оператор ПДн (ТЗ BLOCKER-LEGAL-1 закрыт). Цель — публичный запуск `meno.nsu.ru` с минимальным трением.

---

## 1. Ключевые решения (дельты к ТЗ)

- **Гости — гибрид.** Оставляем `guest_sessions` + 256-битный секрет-токен + `X-Guest-Token` для фиксации согласия, удаления и вклада в улучшение при opt-in. **Серверную историю/синхронизацию для гостя НЕ строим** — история гостя остаётся в localStorage.
- **Единое правило хранения** (см. §5).
- **Рез №2:** документы — версионированный статичный контент (`legal/*.ru.md`) + `version`/`sha256` в конфиге. Таблицы `legal_documents` нет.
- **Рез №3:** retention — CLI на общем deletion-service (cron на хосте НГУ). Без systemd-timer и `--dry-run`-by-default: удаление только с явным `--apply`.
- **Отложено (ТЗ §18):** перенос JWT в HttpOnly-cookie, email-verification, восстановление пароля, SSO, автоперенос гостевой истории в аккаунт, admin UI, публичный рейтинг.

## 2. Инвентарь данных

Согласован с владельцем 2026-07-21. См. таблицу инвентаря в истории обсуждения / `legal/privacy-policy.ru.md` §5–§7, §12 (совпадает).

## 3. Целевая схема БД (Alembic, без `create_all` для основной БД)

### Новые таблицы

**`guest_sessions`**
```
id            str PK
secret_hash   str unique not null      # хранится только хэш; сырой токен — один раз клиенту
created_at    timestamptz not null
last_seen_at  timestamptz not null
expires_at    timestamptz not null
```

**`consent_events`** (append-only)
```
id                str PK
user_id           nullable FK users.id           ON DELETE SET NULL
guest_session_id  nullable FK guest_sessions.id  ON DELETE SET NULL
purpose           str   # SERVICE_AND_HISTORY | ACCOUNT_REGISTRATION | MENO_IMPROVEMENT
action            str   # granted | revoked
document_kind     str   # privacy_policy | personal_data_consent | terms_of_use
document_version  str
document_sha256   str
source            str   # first_run_modal | registration | privacy_settings
created_at        timestamptz not null
CHECK: ровно один из (user_id, guest_session_id) задан
```
IP/User-Agent не сохраняем (ТЗ §6.4).

### Изменения существующих

**`conversations`** — добавить:
```
guest_session_id  nullable FK guest_sessions.id ON DELETE CASCADE
analysis_allowed  bool not null default false
title             nullable str
last_message_at   timestamptz
```
- `user_id`: сделать **настоящим FK** `users.id ON DELETE CASCADE`. ⚠️ Сейчас `String(128)` без FK; при миграции выверить тип относительно `users.id` (проверить в Этапе 1) и очистить/перенести legacy-значения, где `user_id` фактически был session-строкой.
- CHECK: ровно один владелец (`user_id` XOR `guest_session_id`) для непустых/новых диалогов; бесхозный production-диалог создать нельзя.

**`pipeline_runs`** — добавить `conversation_id` FK `conversations.id ON DELETE CASCADE`. Это **ключевой фикс каскада**: сейчас связь только строкой `session_id` без FK, поэтому удаление диалога не достаёт `pipeline_runs`/`generation_records`/`sources`/`pipeline_stage_runs`. Backfill `conversation_id` из `session_id`, где однозначно.

**`message_feedback`, `session_surveys`, `arena_votes`** — привязать к `conversation_id` (FK, CASCADE) напрямую или через `run→conversation`. Для `arena_votes`: агрегат (`model_a/b`,`winner`) хранить всегда; `question`/`response_a`/`response_b` — только при `analysis_allowed=true`, при удалении диалога — удалить текстовую часть.

**`users`** — структурно без изменений (email-логин + bcrypt сохраняем). Правка в коде: пароль >72 UTF-8 байт → понятная ошибка (не тихая обрезка); min 8.

## 4. API

| Метод/путь | Доступ | Назначение |
|---|---|---|
| `POST /v1/guest/session` | — | выдать `{guest_session_id, guest_token, expires_at}`; далее `X-Guest-Token` |
| `GET /v1/legal/documents` | — | актуальные `{kind, version, sha256, url, effective_at}` |
| `GET /v1/privacy/settings` | JWT \| guest-token | текущее состояние согласий субъекта |
| `PATCH /v1/privacy/settings` | JWT \| guest-token | toggle `meno_improvement` → append-only consent event; валидировать `document_version` |
| `GET /v1/conversations` | JWT | пагинированный список истории аккаунта |
| `GET /v1/conversations/{id}` | JWT | диалог аккаунта (гостю недоступно) |
| `DELETE /v1/conversations/{id}` | JWT \| guest-token | удалить свой диалог (каскад) |
| `DELETE /v1/conversations` | JWT \| guest-token | JWT: вся история; guest: все серверные данные этого токена |
| `DELETE /v1/auth/me` | JWT | тело `{password, confirmation:"DELETE"}` → 204 |
| `POST /v1/chat/completions` | JWT \| guest-token | + явный `conversation_id`; проверка владельца; без `SERVICE_AND_HISTORY` → `consent_required` |
| `POST /v1/chat/completions/clear_history` | JWT \| guest-token | deprecated-обёртка: авторизация + владелец + новый deletion-service |

**Правила владения:** JWT видит только `user_id=current`; guest-token — только свой `guest_session_id`; конфликт владельца → **404** (не раскрывать чужой ID). `payload.user` больше не является доказательством владения.

## 5. Правило хранения (минимизация)

| Кто | Что пишем на сервер |
|---|---|
| Гость без opt-in | только обезличенная телеметрия (request id, модель, длительность, длина ответа, код ошибки, ts) |
| Гость с opt-in | `conversation`+`messages`+расширенные записи, `analysis_allowed=true`, привязка к `guest_session_id`; наружу не отдаём |
| Зарегистрированный | `conversation`+`messages` всегда; расширенные записи (`generation_records`, тексты стадий, `search_queries`, arena-тексты) — только при `analysis_allowed=true` |

При `analysis_allowed=false`: `PipelineRun.user_question=NULL`, `search_queries` не пишем, `GenerationRecord` не создаём, stage detail без текстов, failure-record без вопроса, arena question/responses не пишем. Feedback-комментарий — только если пользователь явно отправил (с коротким пояснением, что он доступен команде).

## 6. Согласия

- Первый экран (блокирующая панель до первого сообщения): две кнопки. «Разрешить улучшение и продолжить» → `SERVICE_AND_HISTORY`+`MENO_IMPROVEMENT`; «Продолжить без улучшения» → только `SERVICE_AND_HISTORY`. Вторая кнопка видима, без пред-отмеченных чекбоксов.
- Регистрация: отдельный неотмеченный чекбокс `ACCOUNT_REGISTRATION`; без него регистрация невозможна; улучшение не является условием регистрации.
- Раздел «Данные и конфиденциальность»: toggle улучшения, удалить историю, удалить аккаунт (для гостя — удалить данные браузера), ссылки на 3 документа.
- Версия/хэш документа берутся из конфига; неизвестную/устаревшую версию бэкенд отклоняет.

## 7. Удаление и retention

- **Deletion-service** (транзакционный): `Conversation → messages, pipeline_runs → (stage_runs, sources, generation_records, feedback), arena-тексты, копия в trace-store`. Каскад через FK + явная зачистка не-FK хранилищ (trace-store отдельной БД).
- **Удаление стирает данные и из аналитического контура.** Расширенные записи привязаны к диалогу по FK и удаляются тем же каскадом — отдельной «аналитической копии», переживающей удаление, не создаём. Обоснование: по 152-ФЗ согласие отзывно, а удаление = отзыв + требование стирания; удержание идентифицируемых диалогов удалённого субъекта противоречит и закону, и тексту нашего же согласия/политики (которые обещают, что удаление стирает данные).
- **Отзыв улучшения без удаления ≠ удаление.** При отзыве `analysis_allowed=false`: диалоги исключаются из будущего анализа/выгрузок, но НЕ удаляются (пользователь не просил) и остаются в истории субъекта. Большинство пользователей не удаляют аккаунт — их opt-in диалоги живут в анализе, пока жив диалог.
- **Что переживает удаление:** полностью обезличенные агрегаты/метрики (счётчики, показатели ошибок/качества, arena Elo, кластеры частых вопросов), считаемые rollup-джобом, пока диалог существует. Это не ПДн — хранятся без ограничения (обещано в политике §13). Текст-уровневые де-идентифицированные benchmark-экземпляры — только при реальной анонимизации на этапе захвата; больше работы + риск реидентификации → в v1 не делаем.
- **Account deletion:** проверка пароля → удалить все диалоги пользователя и связанные записи → отвязать/удалить feedback/arena → удалить пользователя → 204.
- **Retention CLI** (`meno-rag retention`): выбирает просроченное по срокам §12; по умолчанию только считает/показывает; удаляет с `--apply`; пакетно, идемпотентно, без содержимого в логах; cron на хосте НГУ.

## 8. Безопасность (в основном багфиксы)

- Фронт: убрать `bodyPreview` (**сделано**), `crypto.randomUUID()` вместо `Math.random()`, logout чистит account-стейт, не логировать email/пароль, гейт логгера в прод-сборке.
- Бэкенд: CORS → `https://meno.nsu.ru` в проде (не wildcard); rate-limit `login`/`register`(+chat); одинаковая ошибка для неверного email/пароля; проверка владельца на каждом read/update/delete; не возвращать email в публичных ответах, никогда — хэш пароля; `OPENROUTER_API_KEY` пуст и `CAPTURE_PIPELINE_TRACE=false` в проде.

## 9. Production safety checks (ТЗ §17)

При `APP_ENV=production` бэкенд отказывается стартовать / кидает критическую ошибку при: SQLite вместо PostgreSQL; пустом/слабом JWT-секрете; wildcard CORS; включённом OpenRouter; включённом trace без явного override+срока; отсутствии актуальной версии legal-документа; отсутствии URL privacy/consent/terms; отсутствии настроек retention.

## 10. Юридические документы

- Каноника: `legal/privacy-policy.ru.md`, `legal/personal-data-consent.ru.md`, `legal/terms-of-use.ru.md`.
- Бэкенд считает `sha256` от файла + берёт `version`/`effective_at`/`url` из конфига; отдаёт `GET /v1/legal/documents`.
- Фронт рендерит по `/privacy`,`/consent`,`/terms` (сейчас роутинга нет — добавить минимальный роутинг **или** отдавать страницы бэкендом за nginx). Footer-ссылки на desktop и mobile + «Данные и конфиденциальность».
- Одноразовое уведомление о localStorage (ТЗ §10.6).

## 11. Поэтапный план

- **Этап 0 — Аудит.** Выполнен (read-only обоих репо); оформить как data-inventory + список расхождений.
- **Этап 1 — Владение и идентичность.** `guest_sessions`+токен; `conversation_id` явным; owner-колонки+FK; каскад-FK на pipeline/generation/feedback/arena; `crypto.randomUUID()`; `clear_history` с авторизацией+владельцем+deletion-service; багфиксы (bodyPreview✓, пароль 72Б, logout). Тесты IDOR.
- **Этап 2 — Согласия и документы.** `consent_events`; version/hash в конфиге; `GET /v1/legal/documents`, `GET/PATCH /v1/privacy/settings`; панель первого экрана; чекбокс регистрации; раздел «Данные и конфиденциальность»; рендер 3 документов + footer.
- **Этап 3 — Политика хранения.** `analysis_allowed` + правило §5; минимальная запись при opt-out, расширенная при opt-in; безопасный CLI-export (по умолчанию только `analysis_allowed=true`).
- **Этап 4 — История и удаление.** `GET /v1/conversations(/{id})`, загрузка при логине, кросс-девайс; удаление чата/истории/аккаунта; guest-удаление; каскад.
- **Этап 5 — Retention/логи/безопасность/ops.** retention CLI; CORS/rate-limit/safety-checks; отчёт nginx/backup/trace; deploy-guide.
- **Этап 6 — Миграция и rollout.** backup → Alembic → backfill `analysis_allowed=false` для legacy → staging smoke → деплой → пост-деплой проверка.

После каждого этапа — зелёные тесты до перехода к следующему.

## 12. Тестирование (ТЗ §15)

Unit: генерация/хэш/верификация guest-токена, разрешение текущего согласия, валидация version/hash, лимит 72 байта, предикаты владения, выборка retention, редакция логов, ветка persistence. Integration: 15 сценариев ТЗ §15.2 (consent_required, first-run события, opt-out без GenerationRecord, opt-in, отзыв, IDOR guest/user, payload.user не захватывает чужое, каскадное удаление, unknown-версия отклонена, OpenRouter/trace off). Frontend: панель до первого сообщения, доступность обеих кнопок, сохранение согласия, logout, кросс-девайс, отсутствие пароля/почты в console, footer. Security regression: IDOR, rate-limit, CORS, отсутствие секретов в ответах/логах, удалённые данные отсутствуют в основной БД/trace/export.

## 13. Плейсхолдеры и блокеры запуска

**Плейсхолдеры документов (нужны реквизиты НГУ):**
- `[[PLACEHOLDER: email для обращений субъектов ПДн]]`
- `[[PLACEHOLDER: ответственный за обработку ПДн / подразделение]]`
- `[[PLACEHOLDER: ИНН/ОГРН НГУ]]` — включить при необходимости
- `[[PLACEHOLDER: дата публикации / вступления в силу]]` (×3 документа)

**Блокеры запуска (ТЗ §3, §17):**
- BLOCKER-INFRA-1 — проверить фактические nginx / backup / Redis / прод `.env` / ротацию логов / `CAPTURE_PIPELINE_TRACE` (Этап 5).
- Прод-конфиг должен проходить все production safety checks (§9).
- Проверка миграции на копии production-like БД (Этап 6).

**Открытые инженерные вопросы:**
- Тип `users.id` и стратегия перевода `conversations.user_id` в FK (выяснить в Этапе 1).
- Рендер документов: минимальный роутинг во фронте vs. отдача бэкендом за nginx (решить в Этапе 2).
- Поведение `DELETE /v1/conversations` для гостя: стирает все серверные записи этого токена (при opt-in); локальная история чистится клиентом.
