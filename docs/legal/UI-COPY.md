# UI-COPY — короткие тексты интерфейса, синхронизированные с документами v2.0

Дата: 2026-07-23. Относится к репозиторию **Meno-Web** (`src/i18n.js`, `components/ConsentModal.jsx`, `components/AuthModal.jsx`, `components/SettingsModal.jsx`).
Изменения по этому документу **применены** в Meno-Web (см. `docs/legal/CHANGELOG.md`).

## О cookie-панели

**По результатам аудита cookies не используются** — ни собственные, ни сторонние: их не устанавливают ни приложение, ни Express-сервер, ни nginx. Используется локальное хранилище браузера (localStorage).

**Вместо cookie-панели реализуется отдельное окно согласия на обработку данных**, а состав локального хранилища раскрывается в §11 Политики и одной пояснительной строкой в разделе «Настройки → О сервисе».

Создавать фиктивные категории cookie-согласия («аналитические», «маркетинговые») не нужно и нельзя: необязательных cookies в Сервисе нет.

> **Требует подтверждения ИБ** (вопрос ИБ-25a): решение обойтись без cookie-панели — вывод команды из технического факта, а не согласованная позиция. С учётом ранее высказанного указания о «согласии в виде куки-панели» это должно быть подтверждено отделом ИБ. Если панель всё же потребуется, нужно определить, о чём именно она спрашивает — необязательных cookies нет.

Отдельно: **уведомление о cookies не объединяется с согласием на анализ диалогов** — это разные сущности, и в текущей реализации первого попросту нет.

---

## 1. Окно согласия (`ConsentModal`)

### Что было не так

- Заголовок «История диалогов» не соответствовал содержанию окна (речь идёт о согласии на улучшение).
- Окно **не сообщало**, что оба варианта — и «Продолжить», и «Не сейчас» — фиксируют согласие на серверное хранение диалогов (`service_and_history: true`). Согласие на хранение получалось неинформированным.
- Текст обещал «использование обезличенных версий всех ваших диалогов», хотя обезличивание не реализовано (G-1).
- Не было сказано, что отказ ничего не ограничивает.

### Новые тексты (ru)

| Ключ | Текст |
|---|---|
| `consentModalTitle` | Сохранение и использование диалогов |
| `consentModalStorageNotice` | Ваши диалоги сохраняются на сервере, чтобы вы могли к ним вернуться. Удалить их можно в любой момент в настройках. |
| `consentModalGrantPrefix` | Нажимая «Разрешить», вы отдельно соглашаетесь на использование ваших диалогов — новых и уже сохранённых — для внутреннего улучшения Менона. Для анализа берутся отдельные копии, очищенные от прямых идентификаторов. Подробнее — в |
| `consentModalConsentLink` | Согласии на обработку персональных данных |
| `consentModalGrantSuffix` | . Выбор можно изменить в настройках. |
| `consentModalDeclineNote` | Отказ ни на что не влияет: Менон работает так же, диалоги сохраняются так же. |
| `consentModalPolicyLink` | Политика обработки персональных данных |
| `consentModalDefer` | Не сейчас |
| `consentModalContinue` | Разрешить |

### Новые тексты (en)

| Ключ | Текст |
|---|---|
| `consentModalTitle` | Storing and using your conversations |
| `consentModalStorageNotice` | Your conversations are stored on the server so you can come back to them. You can delete them at any time in settings. |
| `consentModalGrantPrefix` | By choosing “Allow”, you separately consent to your conversations — new and already stored — being used to improve Meno internally. Analysis uses separate copies stripped of direct identifiers. Details in the |
| `consentModalConsentLink` | Personal Data Processing Consent |
| `consentModalGrantSuffix` | . You can change this in settings. |
| `consentModalDeclineNote` | Declining changes nothing: Meno works the same and your conversations are still saved. |
| `consentModalPolicyLink` | Personal Data Processing Policy |
| `consentModalDefer` | Not now |
| `consentModalContinue` | Allow |

> **Условие выпуска.** Фраза «копии, очищенные от прямых идентификаторов» описывает правило §7 Политики. Этот текст **не должен попасть в продакшен раньше**, чем закрыт блокер G-1 (`IMPLEMENTATION-GAPS.md`) — то есть раньше, чем процедура очистки реализована и применяется при формировании аналитических наборов.

---

## 2. Уведомление при регистрации (`AuthModal`)

### Что было не так

Текст «Создавая аккаунт, вы принимаете Условия и Политику» — это **ознакомление**, а не согласие; считать его согласием на обработку прямо запрещено. При этом при регистрации фиксировалось согласие `SERVICE_AND_HISTORY`, а объявленная в коде цель `ACCOUNT_REGISTRATION` не фиксировалась никогда (G-10).

### Новые тексты

| Ключ | ru | en |
|---|---|---|
| `authConsentNoticePrefix` | Создавая аккаунт, вы даёте согласие на обработку адреса электронной почты и никнейма для создания и обслуживания учётной записи. | By creating an account you consent to the processing of your email address and nickname to create and maintain your account. |
| `authConsentNoticeDocs` | Документы: | Documents: |

`authNicknameHint` изменён вместе с запечатыванием публичного рейтинга: «Необязательно. Публично нигде не отображается.» / «Optional. Not shown publicly anywhere.» — прежний текст обещал показ никнейма в рейтинге участников, до которого у пользователя больше нет доступа (вкладка скрыта, маршрут не смонтирован).

Ссылки под уведомлением: «Согласие на обработку персональных данных», «Пользовательское соглашение», «Политика конфиденциальности» (существующие ключи `consentReadConsent`, `consentReadTerms`, `consentReadPrivacy`), перечисленные через запятую — так снимается проблема падежного согласования и исчезает конструкция «принимаете … Политика конфиденциальности».

Ключ `authConsentNoticeAnd` удалён.

**Остаётся к реализации (G-10):** отдельная фиксация согласия по цели `ACCOUNT_REGISTRATION` при регистрации — сейчас записывается только `SERVICE_AND_HISTORY`.

---

## 3. Раздел «Данные и конфиденциальность» (`SettingsModal`)

### Что было не так

- Подпись переключателя улучшения («Тексты сообщений сохраняются и используются для улучшения сервиса») создавала впечатление, что **само хранение** зависит от переключателя. Хранение зависит от согласия на сервисную обработку и происходит в любом случае.
- «История на этом устройстве» → «Очистить» удаляет только `localStorage`; подпись не подчёркивала, что серверные диалоги остаются (G-15).
- Подсказка удаления обещала удалить и записи о согласиях. Это обещание снято: запись о согласии — доказательство того, что согласие получено (ч. 1 ст. 9 152-ФЗ), она переживает удаление аккаунта и содержит только цель, решение, версию и хэш документа, источник, дату и время. Текст теперь говорит об этом прямо.
- Не было отдельного действия «удалить всю историю на сервере, сохранив аккаунт» — между «очистить этот браузер» и «удалить всё» зияла дыра. Добавлена отдельная строка «История на сервере».

### Новые тексты

| Ключ | ru | en |
|---|---|---|
| `privacyImprovementLabel` | Улучшать Менон на моих диалогах | Use my conversations to improve Meno |
| `privacyImprovementHint` | Диалоги сохраняются в любом случае. Переключатель разрешает использовать их копии, очищенные от прямых идентификаторов, для внутреннего улучшения — как новые диалоги, так и уже сохранённые. | Conversations are stored either way. This switch allows copies stripped of direct identifiers to be used for internal improvement — both new and already stored conversations. |
| `privacyClearLabel` | История в этом браузере | History in this browser |
| `privacyClearHint` | Удаляет только локальную копию. Диалоги на сервере останутся — их удаляет «История на сервере». | Removes the local copy only. Server-side conversations stay — “Server-side history” removes those. |
| `privacyServerHistoryLabel` | История на сервере | Server-side history |
| `privacyServerHistoryHint` | Удаляет все ваши диалоги на сервере. Аккаунт сохраняется. | Deletes all your conversations on the server. Your account is kept. |
| `privacyServerHistoryButton` | Удалить историю | Delete history |
| `privacyServerHistoryConfirm` | Точно удалить | Yes, delete |
| `privacyDeleteHint` | Удаляет аккаунт (если есть) и все диалоги на сервере. Необратимо. Запись о том, какое согласие и когда вы дали, сохраняется как подтверждение — без email и без текста диалогов. | Deletes your account (if any) and all server-side conversations. Irreversible. A record of which consent you gave and when is kept as proof — without your email and without conversation text. |
| `settingsStorageNote` | Сервис не использует cookies: настройки, локальная история и токены хранятся в локальном хранилище браузера. | This service uses no cookies: settings, local history and tokens are kept in the browser’s local storage. |

`settingsStorageNote` — новая строка, выводится в конце раздела «О сервисе», под ссылками на документы.

---

## 4. Что осталось несинхронизированным (требует доработки кода)

| № | Что | Блокер |
|---|---|---|
| 1 | Повторный запрос согласия при смене версии документа — гейт смотрит только на локальный флаг `meno.consentDecided` | G-11 |
| 2 | Раздельная фиксация согласия по цели «учётная запись» (`ACCOUNT_REGISTRATION`) | G-10 |
| 3 | Подтверждение ИБ, что cookie-панель не требуется | ИБ-25a |
