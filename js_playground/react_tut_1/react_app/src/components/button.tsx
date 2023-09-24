interface Props {
  children: string;
  color?: "primary" | "secondary" | "danger" | "success"; // Optional due to having default value
  onClick: () => void;
}

const Button = ({ children, color = "primary", onClick }: Props) => {
  return (
    <button className={"btn btn-" + color} onClick={onClick}>
      {children}
    </button>
  );
};

export default Button;
