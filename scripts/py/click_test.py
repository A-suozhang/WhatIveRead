import click

@click.command()
@click.option('--name', default = "Nobody" , help = "The Name 2 Print")
@click.option('--age', type = int, help = "The Age")
@click.option('--gender', default = "?", type = click.Choice(["Male","FeMale","?"]))
def hello(name, age, gender):
    print("Hello there, {} {} {}".format(name,age, gender))
    click.echo("Click Triggered..")


if __name__ == "__main__":
    hello()
