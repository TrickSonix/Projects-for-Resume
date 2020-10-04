# Решение для проекта "Рекомендательная система для ритейла" для курса GeekBrains
## Постановка задачи
Необходимо было предоставить 5 рекомендаций для каждого пользователя в тестовом датасете.
Эти рекомендации должны были удовлетворять следующим бизнес-требованиям:
- Все товары из разных категорий (sub_commodity_desc)
- 1 дорогой товар, > 7 долларов
- 2 новых товара (юзер никогда не покупал)
## Решение
Задача решалась с помощью построения двухуровневой рекомендательнной системы. Модель певрого уровня: ALS, модель второго уровня: CatBoostClassifier. 
На первом уровне товары отфильтровывались до 6000 самых популярных, из них выбиралось 200 рекомендаций для модели второго уровня.
## Метрика
Целью проекта было достигнуть значения метрики money precision@5 > 20%
## Для запуска проекта
- Скопировать данные о продажах и тестовый датасет(если он есть) в директорию data
- Создать конфигурационный файл src/settings.py
```
python Run_project.py
```
## Результат
В результате было достигнуто значение метрики money precision@5=0.22