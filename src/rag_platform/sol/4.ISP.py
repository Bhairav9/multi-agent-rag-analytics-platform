#4.Interface Segregation Principle

from abc import ABC, abstractmethod

class Machine:
    def print(self, document):
        raise NotImplementedError

    def fax(self, document):
            raise NotImplementedError

    def scan(self, document):
            raise NotImplementedError

class MultiFunctionPrinter(Machine):
    def print(self, document):
        raise NotImplementedError

    def fax(self, document):
            raise NotImplementedError

    def scan(self, document):
            raise NotImplementedError

class OldFashionedPrinter(Machine):
    def print(self, document):
        pass #ok

    def fax(self, document):
            pass #no-op

    def scan(self, document):
            raise NotImplementedError("Printer not support")

class Printer:
    @abstractmethod
    def print(self,document):
        pass

class Scanner:
    @abstractmethod
    def scan(self, document):
        pass

class MyPrinter(Printer):
    def print(self,document):
        print(document)

class MultiFunctionDevice(Printer,Scanner):
    @abstractmethod
    def print(self,document):
        pass

    @abstractmethod
    def scan(self, document):
        pass

class MultiFunctionMachine(MultiFunctionDevice):
    def __init__(self, printer, scanner):
        self.printer=printer
        self.scanner=scanner

    def print(self,document):
        self.printer.print(document)
    
    def scan(self, document):
        self.scanner.scan(document)