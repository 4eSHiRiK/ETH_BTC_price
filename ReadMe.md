Задание:

1. Вам нужно определить собственные движения цены фьючерса ETHUSDT, исключив из них движения вызванные влиянием цены BTCUSDT. Опишите, какую методику вы выбрали, какие параметры подобрали, и почему.

2. Напишите программу на Python, которая в реальном времени (с минимальной задержкой) следит за ценой фьючерса ETHUSDT и используя выбранный вами метод, определяет собственные движение цены ETH. При изменении цены на 1% за последние 60 минут, программа выводит сообщение в консоль. При этом программа должна продолжать работать дальше, постоянно считывая актуальную цену.


Решение:

Для данной задачи я применил линейную регрессию и использовал метод train_test_split библиотеки Scikit-learn. В качестве исходных данных для обучения модели использовал csv файл цен ETHUSDT и BTCUSDT за прошлые года. 

Ссылка на GitHub: https://github.com/4eSHiRiK/ETH_BTC_price